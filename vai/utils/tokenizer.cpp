#include "tokenizer.h"
#include "tiny-json.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_map>


namespace vai::utils {

namespace tokenizer_detail {


/////////////////////////////////////////////////////////////////////////////////////////
// UTF-8
/////////////////////////////////////////////////////////////////////////////////////////
inline void utf8Append(std::string& out, char32_t cp)
{
    if (cp < 0x80) {
        out += char(cp);
    } else if (cp < 0x800) {
        out += char(0xC0 | (cp >> 6));
        out += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += char(0xE0 | (cp >> 12));
        out += char(0x80 | ((cp >> 6) & 0x3F));
        out += char(0x80 | (cp & 0x3F));
    } else {
        out += char(0xF0 | (cp >> 18));
        out += char(0x80 | ((cp >> 12) & 0x3F));
        out += char(0x80 | ((cp >> 6) & 0x3F));
        out += char(0x80 | (cp & 0x3F));
    }
}

// Allocation-free walk over the code points of a UTF-8 string.
template <typename F>
inline void utf8ForEach(std::string_view s, F&& fn)
{
    size_t i = 0;
    while (i < s.size())
    {
        unsigned char c = (unsigned char)s[i];
        char32_t cp;
        size_t len;
        if      (c < 0x80)         { cp = c;        len = 1; }
        else if ((c >> 5) == 0x06) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0x0E) { cp = c & 0x0F; len = 3; }
        else if ((c >> 3) == 0x1E) { cp = c & 0x07; len = 4; }
        else                       { cp = c;        len = 1; }

        if (i + len > s.size())
            len = 1;
        for (size_t k = 1; k < len; ++k)
            cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);

        fn(cp);
        i += len;
    }
}

// Decodes to code points, and records the byte offset each one starts at so a
// match over code points can be sliced back out of the original string.
// offsets ends up one longer than cps: the last entry is the end of the input.
inline void utf8Decode(std::string_view s, std::vector<char32_t>& cps, std::vector<size_t>& offsets)
{
    cps.clear();
    offsets.clear();
    size_t i = 0;
    while (i < s.size())
    {
        offsets.push_back(i);
        unsigned char c = (unsigned char)s[i];
        char32_t cp;
        size_t len;
        if      (c < 0x80) { cp = c;          len = 1; }
        else if ((c >> 5) == 0x06) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0x0E) { cp = c & 0x0F; len = 3; }
        else if ((c >> 3) == 0x1E) { cp = c & 0x07; len = 4; }
        else { cp = c; len = 1; }   // invalid lead byte: pass it through as-is

        if (i + len > s.size())
            len = 1;
        for (size_t k = 1; k < len; ++k)
            cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);

        cps.push_back(cp);
        i += len;
    }
    offsets.push_back(s.size());
}


/////////////////////////////////////////////////////////////////////////////////////////
// BPE model
/////////////////////////////////////////////////////////////////////////////////////////
class BpeModel
{
public:
    using SymbolId = int32_t;
    static constexpr SymbolId invalidSymbol = -1;

    // The vocabulary lives here alone. Every piece string is stored once, as the
    // key of symbolToId_, and everything else refers to it by SymbolId.
    SymbolId registerToken(std::string_view piece, int tokenId)
    {
        const SymbolId symbol = intern(piece);
        tokenIds_[size_t(symbol)] = tokenId;
        ++tokenCount_;
        return symbol;
    }

    int tokenId(SymbolId symbol) const
    {
        return symbol == invalidSymbol ? -1 : tokenIds_[size_t(symbol)];
    }

    // invalidSymbol when the vocabulary has never seen this spelling.
    SymbolId symbolId(std::string_view piece) const
    {
        auto it = symbolToId_.find(piece);
        return it == symbolToId_.end() ? invalidSymbol : it->second;
    }

    std::string_view symbolText(SymbolId symbol) const
    {
        return symbol == invalidSymbol ? std::string_view() : *symbolText_[size_t(symbol)];
    }

    size_t tokenCount() const { return tokenCount_; }

    // Walks the pieces that are vocabulary entries, skipping the symbols that
    // exist only as the result of a merge rule.
    template <typename F>
    void forEachToken(F&& fn) const
    {
        for (SymbolId s = 0; s < nextSymbolId_; ++s)
            if (tokenIds_[size_t(s)] >= 0)
                fn(std::string_view(*symbolText_[size_t(s)]), tokenIds_[size_t(s)]);
    }

    void addMerge(std::string_view left, std::string_view right, int rank)
    {
        const SymbolId leftId  = intern(left);
        const SymbolId rightId = intern(right);

        std::string merged;
        merged.reserve(left.size() + right.size());
        merged.append(left.data(), left.size());
        merged.append(right.data(), right.size());
        const SymbolId resultId = intern(merged);

        merges_.emplace(pairKey(leftId, rightId), Merge{ rank, resultId });
    }

    struct SegmentScratch
    {
        struct Node {
            SymbolId symbol = invalidSymbol;
            size_t begin = 0;
            size_t end = 0;
            int prev = -1;
            int next = -1;
            uint32_t generation = 0;
            bool alive = true;
        };

        struct Candidate {
            int rank = 0;
            int left = -1;
            int right = -1;
            uint32_t leftGeneration = 0;
            uint32_t rightGeneration = 0;
            SymbolId result = invalidSymbol;
        };

        std::vector<char32_t> cps;
        std::vector<size_t> offsets;
        std::vector<Node> nodes;
        std::vector<Candidate> heap;
    };

    // Splits `text` into the symbols left after all ranked BPE merges. Scratch
    // storage is owned by the caller so repeated words reuse the same capacity.
    template <typename F>
    void segment(std::string_view text, SegmentScratch& scratch, F&& emit) const
    {
        if (text.empty())
            return;

        utf8Decode(text, scratch.cps, scratch.offsets);
        if (scratch.cps.empty())
            return;

        using Node = SegmentScratch::Node;
        using Candidate = SegmentScratch::Candidate;

        auto later = [](const Candidate& a, const Candidate& b) {
            if (a.rank != b.rank) return a.rank > b.rank;
            return a.left > b.left;
        };

        auto& nodes = scratch.nodes;
        nodes.clear();
        nodes.resize(scratch.cps.size());
        for (size_t i = 0; i < scratch.cps.size(); ++i)
        {
            auto it = singleCodepointSymbols_.find(scratch.cps[i]);
            nodes[i].symbol = it == singleCodepointSymbols_.end() ? invalidSymbol : it->second;
            nodes[i].begin = scratch.offsets[i];
            nodes[i].end = scratch.offsets[i + 1];
            nodes[i].prev = i == 0 ? -1 : int(i - 1);
            nodes[i].next = i + 1 == scratch.cps.size() ? -1 : int(i + 1);
        }

        auto& heap = scratch.heap;
        heap.clear();

        auto pushPair = [&](int left) {
            if (left < 0 || !nodes[left].alive)
                return;
            const int right = nodes[left].next;
            if (right < 0 || !nodes[right].alive)
                return;
            if (nodes[left].symbol == invalidSymbol || nodes[right].symbol == invalidSymbol)
                return;

            auto it = merges_.find(pairKey(nodes[left].symbol, nodes[right].symbol));
            if (it == merges_.end())
                return;

            heap.push_back(Candidate{ it->second.rank, left, right,
                                      nodes[left].generation, nodes[right].generation,
                                      it->second.result });
            std::push_heap(heap.begin(), heap.end(), later);
        };

        for (size_t i = 0; i + 1 < nodes.size(); ++i)
            pushPair(int(i));

        while (!heap.empty())
        {
            std::pop_heap(heap.begin(), heap.end(), later);
            Candidate candidate = heap.back();
            heap.pop_back();

            Node& left = nodes[candidate.left];
            Node& right = nodes[candidate.right];
            if (!left.alive || !right.alive ||
                left.next != candidate.right || right.prev != candidate.left ||
                left.generation != candidate.leftGeneration ||
                right.generation != candidate.rightGeneration)
                continue;

            left.symbol = candidate.result;
            left.end = right.end;
            ++left.generation;

            right.alive = false;
            ++right.generation;
            left.next = right.next;
            if (right.next >= 0)
                nodes[right.next].prev = candidate.left;

            pushPair(left.prev);
            pushPair(candidate.left);
        }

        int node = 0;
        while (node >= 0)
        {
            const Node& current = nodes[node];
            if (current.alive)
                emit(current.symbol, text.substr(current.begin, current.end - current.begin));
            node = current.next;
        }
    }

private:
    struct Merge {
        int rank = 0;
        SymbolId result = invalidSymbol;
    };

    static uint64_t pairKey(SymbolId left, SymbolId right)
    {
        return (uint64_t(uint32_t(left)) << 32) | uint32_t(right);
    }

    static bool decodeSingleCodepoint(std::string_view text, char32_t& cp)
    {
        if (text.empty())
            return false;

        const unsigned char c = static_cast<unsigned char>(text[0]);
        size_t len = 0;
        if      (c < 0x80)         { cp = c;        len = 1; }
        else if ((c >> 5) == 0x06) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0x0E) { cp = c & 0x0F; len = 3; }
        else if ((c >> 3) == 0x1E) { cp = c & 0x07; len = 4; }
        else                       { cp = c;        len = 1; }

        if (len != text.size())
            return false;
        for (size_t i = 1; i < len; ++i)
            cp = (cp << 6) | (static_cast<unsigned char>(text[i]) & 0x3F);
        return true;
    }

    SymbolId intern(std::string_view symbol)
    {
        // This allocation happens only while tokenizer.json is loaded. The hot
        // encode path uses integer ids exclusively.
        auto it = symbolToId_.find(symbol);
        if (it != symbolToId_.end())
            return it->second;

        const SymbolId id = nextSymbolId_++;
        // Node-based, so the key stays put and symbolText_ can point at it.
        auto placed = symbolToId_.emplace(std::string(symbol), id).first;
        symbolText_.push_back(&placed->first);
        tokenIds_.push_back(-1);

        char32_t cp = 0;
        if (decodeSingleCodepoint(symbol, cp))
            singleCodepointSymbols_.emplace(cp, id);
        return id;
    }

    // Lets find() take a string_view without building a std::string for it.
    struct StringHash
    {
        using is_transparent = void;
        size_t operator()(std::string_view s) const { return std::hash<std::string_view>{}(s); }
    };

    SymbolId nextSymbolId_ = 0;
    size_t   tokenCount_ = 0;
    std::unordered_map<std::string, SymbolId, StringHash, std::equal_to<>> symbolToId_;
    std::vector<const std::string*> symbolText_;   // symbol -> its key in symbolToId_
    std::vector<int> tokenIds_;                    // symbol -> token id, -1 when not a token
    std::unordered_map<char32_t, SymbolId> singleCodepointSymbols_;
    std::unordered_map<uint64_t, Merge> merges_;
};


/////////////////////////////////////////////////////////////////////////////////////////
// Unicode property classes
/////////////////////////////////////////////////////////////////////////////////////////
struct CpRange { char32_t lo, hi; };

inline bool inRanges(char32_t c, const CpRange* r, size_t n)
{
    // The tables are sorted and non-overlapping, so a binary search avoids
    // walking dozens of Unicode ranges for every regex property test.
    size_t lo = 0, hi = n;
    while (lo < hi)
    {
        const size_t mid = lo + (hi - lo) / 2;
        if (c < r[mid].lo)
            hi = mid;
        else if (c > r[mid].hi)
            lo = mid + 1;
        else
            return true;
    }
    return false;
}

inline bool isUniNumber(char32_t c)
{
    static const CpRange kRanges[] = {
        {0x0030,0x0039}, {0x00B2,0x00B3}, {0x00B9,0x00B9}, {0x00BC,0x00BE},
        {0x0660,0x0669}, {0x06F0,0x06F9}, {0x07C0,0x07C9}, {0x0966,0x096F},
        {0x09E6,0x09EF}, {0x0A66,0x0A6F}, {0x0AE6,0x0AEF}, {0x0B66,0x0B6F},
        {0x0BE6,0x0BEF}, {0x0C66,0x0C6F}, {0x0CE6,0x0CEF}, {0x0D66,0x0D6F},
        {0x0E50,0x0E59}, {0x0ED0,0x0ED9}, {0x0F20,0x0F29}, {0x1040,0x1049},
        {0x17E0,0x17E9}, {0x1810,0x1819}, {0x2070,0x2070}, {0x2074,0x2079},
        {0x2080,0x2089}, {0x2150,0x2182}, {0x2185,0x2189}, {0x2460,0x249B},
        {0x3007,0x3007}, {0x3021,0x3029}, {0xFF10,0xFF19},
    };
    return inRanges(c, kRanges, sizeof(kRanges) / sizeof(kRanges[0]));
}

inline bool isUniLetter(char32_t c)
{
    if (isUniNumber(c))
        return false;   // digit blocks sit inside several of the script ranges below

    static const CpRange kRanges[] = {
        {0x0041,0x005A}, {0x0061,0x007A}, {0x00AA,0x00AA}, {0x00B5,0x00B5},
        {0x00BA,0x00BA}, {0x00C0,0x00D6}, {0x00D8,0x00F6}, {0x00F8,0x02C1},
        {0x02C6,0x02D1}, {0x02E0,0x02E4}, {0x0370,0x0374}, {0x0376,0x0377},
        {0x037A,0x037D}, {0x037F,0x037F}, {0x0386,0x0386}, {0x0388,0x03F5},
        {0x03F7,0x0481}, {0x048A,0x052F}, {0x0531,0x0556}, {0x0561,0x0587},
        {0x05D0,0x05F2}, {0x0620,0x064A}, {0x066E,0x06D3}, {0x06D5,0x06D5},
        {0x06E5,0x06E6}, {0x06EE,0x06EF}, {0x06FA,0x06FC}, {0x0710,0x072F},
        {0x0750,0x077F}, {0x0780,0x07A5}, {0x0904,0x0939}, {0x093D,0x093D},
        {0x0950,0x0950}, {0x0958,0x0961}, {0x0972,0x0980}, {0x0985,0x09B9},
        {0x0A05,0x0A39}, {0x0A85,0x0AB9}, {0x0B05,0x0B39}, {0x0B85,0x0BB9},
        {0x0C05,0x0C39}, {0x0C85,0x0CB9}, {0x0D05,0x0D3A}, {0x0D85,0x0DC6},
        {0x0E01,0x0E30}, {0x0E32,0x0E33}, {0x0E40,0x0E46}, {0x0E81,0x0EB0},
        {0x0F40,0x0F6C}, {0x1000,0x102A}, {0x10A0,0x10FA}, {0x1100,0x11FF},
        {0x1200,0x135A}, {0x13A0,0x13F5}, {0x1401,0x166C}, {0x1681,0x169A},
        {0x16A0,0x16EA}, {0x1700,0x1711}, {0x1780,0x17B3}, {0x1820,0x1877},
        {0x1E00,0x1FFF}, {0x2071,0x2071}, {0x207F,0x207F}, {0x2090,0x209C},
        {0x2C00,0x2CE4}, {0x2D00,0x2D2D}, {0x2D30,0x2D67}, {0x3005,0x3006},
        {0x3031,0x3035}, {0x3041,0x3096}, {0x309D,0x30FF}, {0x3105,0x312F},
        {0x3131,0x318E}, {0x31A0,0x31BF}, {0x3400,0x4DBF}, {0x4E00,0x9FFF},
        {0xA000,0xA48C}, {0xA4D0,0xA4FD}, {0xA500,0xA60C}, {0xA640,0xA66E},
        {0xA717,0xA71F}, {0xA800,0xA822}, {0xA840,0xA873}, {0xA960,0xA97C},
        {0xAC00,0xD7A3}, {0xD7B0,0xD7FB}, {0xF900,0xFAFF}, {0xFB00,0xFB17},
        {0xFB1D,0xFBB1}, {0xFDF0,0xFDFB}, {0xFE70,0xFEFC}, {0xFF21,0xFF3A},
        {0xFF41,0xFF5A}, {0xFF66,0xFFDC},
        {0x10000,0x1000B}, {0x1D400,0x1D7CB}, {0x20000,0x2A6DF}, {0x2A700,0x2EBEF},
    };
    return inRanges(c, kRanges, sizeof(kRanges) / sizeof(kRanges[0]));
}

inline bool isUniSpace(char32_t c)
{
    static const CpRange kRanges[] = {
        {0x0009,0x000D}, {0x0020,0x0020}, {0x0085,0x0085}, {0x00A0,0x00A0},
        {0x1680,0x1680}, {0x2000,0x200A}, {0x2028,0x2029}, {0x202F,0x202F},
        {0x205F,0x205F}, {0x3000,0x3000},
    };
    return inRanges(c, kRanges, sizeof(kRanges) / sizeof(kRanges[0]));
}


/////////////////////////////////////////////////////////////////////////////////////////
// Regex subset used by HuggingFace pre-tokenizer Split patterns
/////////////////////////////////////////////////////////////////////////////////////////
struct RxNode
{
    enum Kind { Lit, Cls, Any, Cat, Alt, Rep, Look };

    // Class members are either a literal code-point range (prop == None) or a
    // property test, and `neg` inverts the whole set.
    enum Prop { None, PropL, PropN, PropSpace, PropNotSpace, PropDigit, PropNotDigit, PropWord, PropNotWord };
    struct Item { char32_t lo = 0, hi = 0; Prop prop = None; };

    Kind kind = Lit;
    char32_t ch = 0;                // Lit
    std::vector<Item> items;        // Cls
    bool neg = false;               // Cls
    bool icase = false;             // Lit / Cls
    int  mn = 0, mx = -1;           // Rep, mx < 0 means unbounded
    bool negLook = false;           // Look
    std::vector<RxNode> kids;
};

inline char32_t lowerAscii(char32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

class Regex
{
    RxNode root_;

public:
    explicit Regex(const std::string& pattern)
    {
        std::vector<char32_t> pat;
        std::vector<size_t>   off;
        utf8Decode(pattern, pat, off);
        size_t pos = 0;
        root_ = parseAlt(pat, pos, false);
        if (pos != pat.size())
            throw std::runtime_error("tokenizer: unparsed trailing regex at offset " + std::to_string(pos));
    }

    // Matches at exactly `pos` (never scans forward) and reports where the match
    // ends. Alternation is ordered and quantifiers are greedy, matching the
    // reference engine's leftmost-first semantics.
    bool matchAt(const std::vector<char32_t>& s, size_t pos, size_t& end) const
    {
        return match(root_, s, pos, [&](size_t e) { end = e; return true; });
    }

private:
    // Non-owning continuation. std::function would construct, and on most standard
    // libraries heap-allocate, at every step of a match. A template parameter is
    // not an option: matchSeq hands itself a new lambda per element, so each
    // nesting level would be a fresh type and the instantiation would not
    // terminate. The lambda always outlives the call it is passed to.
    class Cont
    {
        const void* obj_;
        bool (*call_)(const void*, size_t);

    public:
        Cont(const Cont&) = default;

        template <typename F, typename = std::enable_if_t<!std::is_same_v<std::decay_t<F>, Cont>>>
        Cont(const F& f)
        : obj_(&f)
        , call_([](const void* o, size_t e) { return (*static_cast<const F*>(o))(e); })
        {}

        bool operator()(size_t e) const { return call_(obj_, e); }
    };

    static bool isQuant(char32_t c) { return c == '*' || c == '+' || c == '?' || c == '{'; }

    static RxNode parseAlt(const std::vector<char32_t>& p, size_t& i, bool icase)
    {
        RxNode alt;
        alt.kind = RxNode::Alt;
        alt.kids.push_back(parseCat(p, i, icase));
        while (i < p.size() && p[i] == '|')
        {
            ++i;
            alt.kids.push_back(parseCat(p, i, icase));
        }
        return alt.kids.size() == 1 ? alt.kids[0] : alt;
    }

    static RxNode parseCat(const std::vector<char32_t>& p, size_t& i, bool icase)
    {
        RxNode cat;
        cat.kind = RxNode::Cat;
        while (i < p.size() && p[i] != '|' && p[i] != ')')
            cat.kids.push_back(parseRep(p, i, icase));
        return cat.kids.size() == 1 ? cat.kids[0] : cat;
    }

    static RxNode parseRep(const std::vector<char32_t>& p, size_t& i, bool icase)
    {
        RxNode atom = parseAtom(p, i, icase);
        if (i >= p.size() || !isQuant(p[i]))
            return atom;

        RxNode rep;
        rep.kind = RxNode::Rep;
        if (p[i] == '*')      { rep.mn = 0; rep.mx = -1; ++i; }
        else if (p[i] == '+') { rep.mn = 1; rep.mx = -1; ++i; }
        else if (p[i] == '?') { rep.mn = 0; rep.mx =  1; ++i; }
        else
        {
            size_t save = i;
            ++i;                                  // '{'
            int lo = 0, hi = -1;
            bool ok = i < p.size() && p[i] >= '0' && p[i] <= '9';
            while (i < p.size() && p[i] >= '0' && p[i] <= '9') lo = lo * 10 + int(p[i++] - '0');
            if (i < p.size() && p[i] == ',')
            {
                ++i;
                if (i < p.size() && p[i] >= '0' && p[i] <= '9')
                {
                    hi = 0;
                    while (i < p.size() && p[i] >= '0' && p[i] <= '9') hi = hi * 10 + int(p[i++] - '0');
                }
            }
            else hi = lo;
            if (!ok || i >= p.size() || p[i] != '}')
            {
                i = save;                         // a bare '{' is a literal
                return atom;
            }
            ++i;
            rep.mn = lo; rep.mx = hi;
        }
        if (i < p.size() && (p[i] == '?' || p[i] == '+'))
            throw std::runtime_error("tokenizer: lazy/possessive quantifiers are not supported");
        rep.kids.push_back(std::move(atom));
        return rep;
    }

    static RxNode parseAtom(const std::vector<char32_t>& p, size_t& i, bool icase)
    {
        if (i >= p.size())
            throw std::runtime_error("tokenizer: unexpected end of regex");

        char32_t c = p[i];

        if (c == '(')
        {
            ++i;
            bool look = false, negLook = false, groupIcase = icase;
            if (i < p.size() && p[i] == '?')
            {
                ++i;
                if (i < p.size() && p[i] == ':')      { ++i; }
                else if (i < p.size() && p[i] == '!') { ++i; look = true; negLook = true; }
                else if (i < p.size() && p[i] == '=') { ++i; look = true; }
                else if (i < p.size() && p[i] == 'i')
                {
                    ++i;
                    groupIcase = true;
                    if (i < p.size() && p[i] == ':') ++i;
                    else throw std::runtime_error("tokenizer: only the (?i:...) inline-flag form is supported");
                }
                else throw std::runtime_error("tokenizer: unsupported group modifier in regex");
            }
            RxNode inner = parseAlt(p, i, groupIcase);
            if (i >= p.size() || p[i] != ')')
                throw std::runtime_error("tokenizer: unbalanced '(' in regex");
            ++i;
            if (!look)
                return inner;
            RxNode n;
            n.kind = RxNode::Look;
            n.negLook = negLook;
            n.kids.push_back(std::move(inner));
            return n;
        }

        if (c == '[')
        {
            ++i;
            RxNode n;
            n.kind = RxNode::Cls;
            n.icase = icase;
            if (i < p.size() && p[i] == '^') { n.neg = true; ++i; }
            while (i < p.size() && p[i] != ']')
            {
                RxNode::Item item;
                if (p[i] == '\\') { ++i; item = parseEscape(p, i); }
                else              { item.lo = item.hi = p[i++]; }

                if (item.prop == RxNode::None && i + 1 < p.size() && p[i] == '-' && p[i + 1] != ']')
                {
                    ++i;
                    RxNode::Item hi;
                    if (p[i] == '\\') { ++i; hi = parseEscape(p, i); }
                    else              { hi.lo = hi.hi = p[i++]; }
                    if (hi.prop != RxNode::None)
                        throw std::runtime_error("tokenizer: property class used as a range endpoint");
                    item.hi = hi.hi;
                }
                n.items.push_back(item);
            }
            if (i >= p.size())
                throw std::runtime_error("tokenizer: unbalanced '[' in regex");
            ++i;
            return n;
        }

        if (c == '\\')
        {
            ++i;
            RxNode n;
            n.kind = RxNode::Cls;
            n.icase = icase;
            n.items.push_back(parseEscape(p, i));
            return n;
        }

        if (c == '.')
        {
            ++i;
            RxNode n;
            n.kind = RxNode::Any;
            return n;
        }

        if (c == '*' || c == '+' || c == '?')
            throw std::runtime_error("tokenizer: quantifier without a preceding atom");

        ++i;
        RxNode n;
        n.kind = RxNode::Lit;
        n.ch = c;
        n.icase = icase;
        return n;
    }

    // Consumes one escape sequence and returns it as a class member.
    static RxNode::Item parseEscape(const std::vector<char32_t>& p, size_t& i)
    {
        if (i >= p.size())
            throw std::runtime_error("tokenizer: trailing backslash in regex");

        RxNode::Item item;
        char32_t e = p[i++];
        switch (e)
        {
        case 'p': case 'P': {
            if (i >= p.size() || p[i] != '{')
                throw std::runtime_error("tokenizer: \\p must be followed by {");
            ++i;
            std::string name;
            while (i < p.size() && p[i] != '}') name += char(p[i++]);
            if (i >= p.size())
                throw std::runtime_error("tokenizer: unterminated \\p{...}");
            ++i;
            if      (name == "L" || name == "Letter") item.prop = RxNode::PropL;
            else if (name == "N" || name == "Number") item.prop = RxNode::PropN;
            else throw std::runtime_error("tokenizer: unsupported unicode property \\p{" + name + "}");
            if (e == 'P')
                throw std::runtime_error("tokenizer: negated \\P{...} is not supported");
            return item;
        }
        case 's': item.prop = RxNode::PropSpace;    return item;
        case 'S': item.prop = RxNode::PropNotSpace; return item;
        case 'd': item.prop = RxNode::PropDigit;    return item;
        case 'D': item.prop = RxNode::PropNotDigit; return item;
        case 'w': item.prop = RxNode::PropWord;     return item;
        case 'W': item.prop = RxNode::PropNotWord;  return item;
        case 'n': item.lo = item.hi = '\n'; return item;
        case 'r': item.lo = item.hi = '\r'; return item;
        case 't': item.lo = item.hi = '\t'; return item;
        case 'f': item.lo = item.hi = '\f'; return item;
        case 'v': item.lo = item.hi = '\v'; return item;
        case '0': item.lo = item.hi = '\0'; return item;
        default:  item.lo = item.hi = e;    return item;   // escaped metacharacter
        }
    }

    static bool itemMatches(const RxNode::Item& it, char32_t c)
    {
        switch (it.prop)
        {
        case RxNode::None:         return c >= it.lo && c <= it.hi;
        case RxNode::PropL:        return isUniLetter(c);
        case RxNode::PropN:        return isUniNumber(c);
        case RxNode::PropSpace:    return isUniSpace(c);
        case RxNode::PropNotSpace: return !isUniSpace(c);
        case RxNode::PropDigit:    return c >= '0' && c <= '9';
        case RxNode::PropNotDigit: return !(c >= '0' && c <= '9');
        case RxNode::PropWord:     return isUniLetter(c) || isUniNumber(c) || c == '_';
        case RxNode::PropNotWord:  return !(isUniLetter(c) || isUniNumber(c) || c == '_');
        }
        return false;
    }

    static bool clsMatches(const RxNode& n, char32_t c)
    {
        bool hit = false;
        for (const auto& it : n.items)
        {
            if (itemMatches(it, c) ||
                (n.icase && it.prop == RxNode::None && itemMatches(it, lowerAscii(c))))
            {
                hit = true;
                break;
            }
        }
        return n.neg ? !hit : hit;
    }

    bool match(const RxNode& n, const std::vector<char32_t>& s, size_t pos, const Cont& k) const
    {
        switch (n.kind)
        {
        case RxNode::Lit:
            if (pos >= s.size()) return false;
            if (n.icase ? lowerAscii(s[pos]) != lowerAscii(n.ch) : s[pos] != n.ch) return false;
            return k(pos + 1);

        case RxNode::Any:
            return pos < s.size() && k(pos + 1);

        case RxNode::Cls:
            return pos < s.size() && clsMatches(n, s[pos]) && k(pos + 1);

        case RxNode::Cat:
            return matchSeq(n, 0, s, pos, k);

        case RxNode::Alt:
            for (const auto& kid : n.kids)
                if (match(kid, s, pos, k))
                    return true;
            return false;

        case RxNode::Rep:
            return matchRep(n, 0, s, pos, k);

        case RxNode::Look: {
            bool hit = match(n.kids[0], s, pos, [](size_t) { return true; });
            return (hit != n.negLook) && k(pos);
        }
        }
        return false;
    }

    bool matchSeq(const RxNode& n, size_t idx, const std::vector<char32_t>& s, size_t pos, const Cont& k) const
    {
        if (idx == n.kids.size())
            return k(pos);
        return match(n.kids[idx], s, pos, [&](size_t np) { return matchSeq(n, idx + 1, s, np, k); });
    }

    // True when the body always consumes exactly one code point, which lets a
    // repetition be counted in a loop instead of recursed through.
    static bool isSingleWidth(const RxNode& n)
    {
        return n.kind == RxNode::Lit || n.kind == RxNode::Cls || n.kind == RxNode::Any;
    }

    static bool matchOne(const RxNode& n, const std::vector<char32_t>& s, size_t pos)
    {
        if (pos >= s.size())
            return false;
        switch (n.kind)
        {
        case RxNode::Lit: return n.icase ? lowerAscii(s[pos]) == lowerAscii(n.ch) : s[pos] == n.ch;
        case RxNode::Any: return true;
        case RxNode::Cls: return clsMatches(n, s[pos]);
        default:          return false;
        }
    }

    bool matchRep(const RxNode& n, int count, const std::vector<char32_t>& s, size_t pos, const Cont& k) const
    {
        // Every pattern these files declare repeats a single code point, and
        // recursing once per repetition would put the run length on the stack:
        // a few thousand letters in a row would overflow it. Count the run, then
        // give positions back one at a time to keep the greedy semantics.
        if (isSingleWidth(n.kids[0]))
        {
            int taken = count;
            size_t end = pos;
            while ((n.mx < 0 || taken < n.mx) && matchOne(n.kids[0], s, end)) { ++end; ++taken; }

            for (;; --taken, --end)
            {
                if (taken >= n.mn && k(end))
                    return true;
                if (taken <= count)
                    return false;
            }
        }

        if (n.mx < 0 || count < n.mx)
        {
            // The empty-match guard keeps patterns like (a?)* from looping forever.
            if (match(n.kids[0], s, pos, [&](size_t np) {
                    return np != pos && matchRep(n, count + 1, s, np, k);
                }))
                return true;
        }
        return count >= n.mn && k(pos);
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// One pre-tokenizer stage. A Split pattern matches delimiters, so the pieces are
// the text between matches and the behavior says which piece a match itself joins.
// ByteLevel's implicit pattern instead matches the words themselves, which comes out
// as `isolated` with nothing left between them.
/////////////////////////////////////////////////////////////////////////////////////////
struct SplitStep
{
    enum Behavior { isolated, mergedWithPrevious, mergedWithNext, removed };

    std::unique_ptr<Regex> pattern;
    Behavior behavior = isolated;
};


/////////////////////////////////////////////////////////////////////////////////////////
// GPT-2 byte-level alphabet: every byte gets a printable code point so that BPE
// operates on text while still being able to represent arbitrary bytes.
/////////////////////////////////////////////////////////////////////////////////////////
struct ByteAlphabet
{
    static constexpr char32_t kCpLimit = 0x200;   // every alphabet code point is below this

    std::array<std::string, 256> toPiece;
    std::array<int, kCpLimit>    toByte;          // code point -> byte, -1 when unmapped

    ByteAlphabet()
    {
        toByte.fill(-1);

        std::array<bool, 256> direct{};
        auto mark = [&](int lo, int hi) { for (int b = lo; b <= hi; ++b) direct[b] = true; };
        mark('!', '~'); mark(0xA1, 0xAC); mark(0xAE, 0xFF);

        int extra = 0;
        for (int b = 0; b < 256; ++b)
        {
            char32_t cp = direct[b] ? char32_t(b) : char32_t(256 + extra++);
            utf8Append(toPiece[b], cp);
            toByte[cp] = b;
        }
    }
};

inline const ByteAlphabet& byteAlphabet()
{
    static const ByteAlphabet inst;
    return inst;
}


/////////////////////////////////////////////////////////////////////////////////////////
// Text rewrites: the normalizer runs them over the input before tokenizing, the
// decoder runs them back over each piece. Both pipelines in tokenizer.json are
// sequences of the same two rewrites, so they share a type.
/////////////////////////////////////////////////////////////////////////////////////////
struct TextStep
{
    enum Kind { prepend, replace };

    Kind kind = replace;
    std::string from;   // replace: the literal being matched
    std::string to;     // the text put in its place, or prepended
};

inline void applyTextSteps(const std::vector<TextStep>& steps, std::string& text)
{
    for (const auto& step : steps)
    {
        if (step.kind == TextStep::prepend)
        {
            text.insert(0, step.to);
            continue;
        }
        if (step.from.empty())
            continue;
        for (size_t at = text.find(step.from); at != std::string::npos;
             at = text.find(step.from, at + step.to.size()))
            text.replace(at, step.from.size(), step.to);
    }
}

// "<0xNN>" spells one raw byte in a SentencePiece vocabulary; -1 when the piece
// is ordinary text.
inline int byteFallbackValue(std::string_view piece)
{
    if (piece.size() != 6 || piece.substr(0, 3) != "<0x" || piece[5] != '>')
        return -1;
    auto hex = [](char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    int hi = hex(piece[3]), lo = hex(piece[4]);
    return (hi < 0 || lo < 0) ? -1 : hi * 16 + lo;
}


/////////////////////////////////////////////////////////////////////////////////////////
// How a vocabulary spells text. The decoder the file declares picks one of these
// at load time, and nothing downstream asks again: the choice is an object, not a
// flag consulted at every conversion.
/////////////////////////////////////////////////////////////////////////////////////////
struct PieceCodec
{
    virtual ~PieceCodec() = default;

    // Rewrites one pre-tokenized word into the spelling the merges operate on.
    // Returns a view into `storage` when a rewrite was needed, `word` otherwise.
    virtual std::string_view toPieceText(std::string_view word, std::string& storage) const = 0;

    // Appends one vocabulary piece to `out` as the text it stands for.
    virtual void appendText(std::string_view piece, const std::vector<TextStep>& decodeSteps,
                            std::string& out) const = 0;

    // ByteLevel BPE always splits before merging; SentencePiece merges over whole
    // fragments and normally declares no pre-tokenizer at all.
    virtual bool requiresPreTokenizer() const = 0;

    // Called once the vocabulary is in, to reject a file this codec cannot serve.
    virtual void validate(const BpeModel& vocab, const std::string& path) const = 0;
};


// Every byte was remapped to a printable code point, so a piece is text and the
// byte table inverts it exactly.
struct ByteLevelCodec : PieceCodec
{
    std::string_view toPieceText(std::string_view word, std::string& storage) const override
    {
        storage.clear();
        storage.reserve(word.size() * 2);
        for (unsigned char c : word)
            storage += byteAlphabet().toPiece[c];
        return storage;
    }

    void appendText(std::string_view piece, const std::vector<TextStep>&,
                    std::string& out) const override
    {
        // validate() rejected any piece outside the alphabet, so this resolves.
        const auto& alphabet = byteAlphabet();
        utf8ForEach(piece, [&](char32_t cp) { out += char(alphabet.toByte[cp]); });
    }

    bool requiresPreTokenizer() const override { return true; }

    // appendText maps every vocabulary character straight through the byte table,
    // which only holds if the vocabulary really is byte-level encoded. This catches
    // a damaged file, or a BPE model whose pieces never met a ByteLevel stage.
    void validate(const BpeModel& vocab, const std::string& path) const override
    {
        const auto& alphabet = byteAlphabet();
        vocab.forEachToken([&](std::string_view piece, int id) {
            char32_t bad = 0;
            utf8ForEach(piece, [&](char32_t cp) {
                if (bad == 0 && (cp >= ByteAlphabet::kCpLimit || alphabet.toByte[cp] < 0))
                    bad = cp;
            });
            if (bad != 0)
                throw std::runtime_error(
                    "tokenizer: vocabulary piece \"" + std::string(piece) + "\" (id " +
                    std::to_string(id) + ") holds U+" + std::to_string((uint32_t)bad) +
                    ", which is not a byte-level character — " + path +
                    " is not a ByteLevel BPE tokenizer");
        });
    }
};


// Pieces hold raw text with U+2581 standing in for a space, and the bytes the
// vocabulary cannot spell are carried by <0xNN> tokens.
struct SentencePieceCodec : PieceCodec
{
    std::string_view toPieceText(std::string_view word, std::string&) const override
    {
        return word;   // the merges run over the text exactly as it stands
    }

    void appendText(std::string_view piece, const std::vector<TextStep>& decodeSteps,
                    std::string& out) const override
    {
        const int byte = byteFallbackValue(piece);
        if (byte >= 0) { out += char(byte); return; }

        if (decodeSteps.empty()) { out.append(piece); return; }

        std::string text(piece);
        applyTextSteps(decodeSteps, text);
        out += text;
    }

    bool requiresPreTokenizer() const override { return false; }

    void validate(const BpeModel&, const std::string&) const override {}
};


// A ByteLevel pre-tokenizer splits on its own unless "use_regex": false, using a
// pattern the file never states. HuggingFace's tokenizers library holds it as a
// constant, so it has to be repeated here. GPT-2 introduced it; it is now the
// default for anything that leaves the splitting to ByteLevel.
inline constexpr const char* kByteLevelSplit =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";


/////////////////////////////////////////////////////////////////////////////////////////
// JSON helpers on top of tinyJson
/////////////////////////////////////////////////////////////////////////////////////////
inline std::string jsonString(const tinyjson::sj_Value& v)
{
    std::string out;
    for (char* p = v.start; p != v.end; ++p)
    {
        if (*p != '\\') { out += *p; continue; }
        if (++p == v.end) break;
        switch (*p)
        {
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'u': {
            auto hex4 = [&](char* q) {
                unsigned v4 = 0;
                for (int i = 0; i < 4 && q + i < v.end; ++i)
                {
                    char h = q[i];
                    v4 = v4 * 16 + (h <= '9' ? h - '0' : (h | 0x20) - 'a' + 10);
                }
                return v4;
            };
            if (p + 4 >= v.end) break;
            char32_t cp = hex4(p + 1);
            p += 4;
            if (cp >= 0xD800 && cp <= 0xDBFF && p + 6 < v.end && p[1] == '\\' && p[2] == 'u')
            {
                char32_t lo = hex4(p + 3);
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                p += 6;
            }
            utf8Append(out, cp);
            break;
        }
        default: out += *p; break;   // \" \\ \/
        }
    }
    return out;
}

inline long long jsonInt(const tinyjson::sj_Value& v)
{
    return std::strtoll(v.start, nullptr, 10);
}

inline bool jsonBool(const tinyjson::sj_Value& v)
{
    return v.end - v.start == 4 && v.start[0] == 't';
}


} // namespace tokenizer_detail


/////////////////////////////////////////////////////////////////////////////////////////
// Tokenizer
/////////////////////////////////////////////////////////////////////////////////////////
class Tokenizer::Impl
{
    struct AddedToken { std::string content; bool special = false; };

    // One encode call owns one context. Temporary buffers live here, so the
    // encoding helpers keep their semantic interfaces instead of threading
    // scratch storage through every call.
    struct EncodeScratch
    {
        std::string normalized;
        std::string mapped;
        std::vector<std::string_view> piecesA;
        std::vector<std::string_view> piecesB;
        std::vector<char32_t> splitCps;
        std::vector<size_t> splitOffsets;
        tokenizer_detail::BpeModel::SegmentScratch bpe;
    };

    class EncodeContext
    {
    public:
        EncodeContext(const Impl& tokenizer, std::vector<int>& ids)
        : tokenizer_(tokenizer), ids_(ids)
        {}

        void encodePlain(std::string_view rawText)
        {
            if (rawText.empty())
                return;

            scratch_.normalized.assign(rawText.data(), rawText.size());
            tokenizer_detail::applyTextSteps(tokenizer_.normalizeSteps_, scratch_.normalized);

            auto* pieces = &scratch_.piecesA;
            auto* next = &scratch_.piecesB;
            pieces->clear();
            pieces->push_back(std::string_view(scratch_.normalized));

            for (const auto& step : tokenizer_.splits_)
            {
                next->clear();
                if (next->capacity() < pieces->size() * 2)
                    next->reserve(pieces->size() * 2);
                for (std::string_view piece : *pieces)
                    splitWith(step, piece, *next);
                std::swap(pieces, next);
            }

            for (std::string_view piece : *pieces)
                encodeWord(piece);
        }

    private:
        // Text no match ever claims stays one piece rather than falling apart into
        // single code points, which is what a Split whose delimiter the normalizer
        // already rewrote comes down to.
        void splitWith(const tokenizer_detail::SplitStep& step, std::string_view text,
                       std::vector<std::string_view>& out)
        {
            using Split = tokenizer_detail::SplitStep;

            tokenizer_detail::utf8Decode(text, scratch_.splitCps, scratch_.splitOffsets);
            const auto& cps = scratch_.splitCps;
            const auto& offs = scratch_.splitOffsets;

            auto emit = [&](size_t from, size_t to) {
                if (from < to)
                    out.push_back(text.substr(offs[from], offs[to] - offs[from]));
            };

            size_t piece = 0;
            size_t i = 0;
            while (i < cps.size())
            {
                size_t end = i;
                if (!step.pattern->matchAt(cps, i, end) || end <= i)
                {
                    ++i;
                    continue;
                }

                if (step.behavior == Split::mergedWithNext)
                {
                    emit(piece, i);
                    piece = i;
                }
                else
                {
                    emit(piece, step.behavior == Split::mergedWithPrevious ? end : i);
                    if (step.behavior == Split::isolated)
                        emit(i, end);
                    piece = end;
                }
                i = end;
            }
            emit(piece, cps.size());
        }

        void encodeWord(std::string_view word)
        {
            std::string_view mapped = tokenizer_.codec_->toPieceText(word, scratch_.mapped);

            if (tokenizer_.ignoreMerges_)
            {
                const int whole =
                    tokenizer_.bpeModel_.tokenId(tokenizer_.bpeModel_.symbolId(mapped));
                if (whole >= 0) { ids_.push_back(whole); return; }
            }

            tokenizer_.bpeModel_.segment(
                mapped, scratch_.bpe,
                [&](tokenizer_detail::BpeModel::SymbolId symbol, std::string_view sym) {
                    const int tokenId = tokenizer_.bpeModel_.tokenId(symbol);
                    if (tokenId >= 0) { ids_.push_back(tokenId); return; }

                    if (tokenizer_.byteFallback_)
                    {
                        for (unsigned char c : sym)
                            if (tokenizer_.byteFallbackIds_[c] >= 0)
                                ids_.push_back(tokenizer_.byteFallbackIds_[c]);
                    }
                    else if (tokenizer_.unkId_ >= 0)
                    {
                        ids_.push_back(tokenizer_.unkId_);
                    }
                });
        }

        const Impl& tokenizer_;
        std::vector<int>& ids_;
        EncodeScratch scratch_;
    };

public:
    explicit Impl(const std::string& tokenizerJsonPath) { load(tokenizerJsonPath); }

    std::vector<int> encode(const std::string& text, bool addSpecial, bool parseSpecial) const
    {
        std::vector<int> ids;
        ids.reserve(prefixIds_.size() + suffixIds_.size() + text.size() / 3 + 1);
        EncodeContext context(*this, ids);

        if (addSpecial)
            ids.insert(ids.end(), prefixIds_.begin(), prefixIds_.end());

        const std::string_view input(text);
        size_t plainBegin = 0;
        for (size_t i = 0; parseSpecial && i < input.size(); )
        {
            int id = -1;
            const size_t len = matchAddedToken(input, i, id);
            if (len == 0) { ++i; continue; }

            context.encodePlain(input.substr(plainBegin, i - plainBegin));
            ids.push_back(id);
            plainBegin = i = i + len;
        }
        context.encodePlain(input.substr(plainBegin));

        if (addSpecial)
            ids.insert(ids.end(), suffixIds_.begin(), suffixIds_.end());
        return ids;
    }

    std::string decode(const std::vector<int>& ids, bool skipSpecial) const
    {
        std::string out;

        for (int id : ids)
        {
            auto added = addedById_.find(id);
            if (added != addedById_.end())
            {
                // Added tokens hold raw text rather than encoded pieces, but the
                // decoder's rewrites still reach them: Gemma spells a run of spaces
                // as an added token carrying the U+2581 marker.
                if (!(skipSpecial && added->second.special))
                {
                    std::string piece = added->second.content;
                    tokenizer_detail::applyTextSteps(decodeSteps_, piece);
                    out += piece;
                }
                continue;
            }

            const auto symbol = modelSymbol(id);
            if (symbol == tokenizer_detail::BpeModel::invalidSymbol)
                continue;

            codec_->appendText(bpeModel_.symbolText(symbol), decodeSteps_, out);
        }
        return out;
    }

    // An added token may restate an id the model vocabulary already holds
    // (GPT-2 lists <|endoftext|> in both), so only the new ones are added on.
    int vocabSize() const
    {
        size_t extra = 0;
        for (const auto& [id, tok] : addedById_)
            extra += modelSymbol(id) == tokenizer_detail::BpeModel::invalidSymbol;
        return int(bpeModel_.tokenCount() + extra);
    }

    int pieceToId(const std::string& piece) const
    {
        auto a = addedByContent_.find(piece);
        if (a != addedByContent_.end())
            return a->second;
        return bpeModel_.tokenId(bpeModel_.symbolId(piece));
    }

    std::string idToPiece(int id) const
    {
        auto a = addedById_.find(id);
        if (a != addedById_.end())
            return a->second.content;
        return std::string(bpeModel_.symbolText(modelSymbol(id)));
    }

    bool isSpecial(int id) const
    {
        auto a = addedById_.find(id);
        return a != addedById_.end() && a->second.special;
    }

    const std::vector<int>& prefixIds() const { return prefixIds_; }
    const std::vector<int>& suffixIds() const { return suffixIds_; }

private:
    // Token id -> the model vocabulary's symbol, or invalidSymbol for an id the
    // model does not hold (an added token that is not in the vocabulary).
    tokenizer_detail::BpeModel::SymbolId modelSymbol(int id) const
    {
        if (id < 0 || size_t(id) >= idToSymbol_.size())
            return tokenizer_detail::BpeModel::invalidSymbol;
        return idToSymbol_[size_t(id)];
    }

    // Prefix tree over the added-token strings. Encoding probes it at every byte
    // of the input, so the cost that matters is how fast a position that is *not*
    // an added token gives up: one lookup when no token starts with that byte,
    // and one more per shared character after that. It bounds the per-position
    // work by the longest added token instead of by how many there are.
    struct TrieNode {
        std::unordered_map<unsigned char, int> next;
        int id = -1;    // >= 0 when a token ends here
    };

    tokenizer_detail::BpeModel bpeModel_;
    // Token id -> symbol. Ids are dense, so this is an array rather than a map.
    std::vector<tokenizer_detail::BpeModel::SymbolId> idToSymbol_;
    std::unordered_map<int, AddedToken>  addedById_;
    std::unordered_map<std::string, int> addedByContent_;
    std::array<int, 256>  addedRoot_;      // first byte -> trie node, -1 when no token starts with it
    std::vector<TrieNode> addedTrie_;
    std::vector<int> prefixIds_, suffixIds_;
    std::vector<std::string> prefixNames_, suffixNames_;   // resolved once the vocabulary is in
    std::vector<tokenizer_detail::SplitStep> splits_;
    std::vector<tokenizer_detail::TextStep> normalizeSteps_;   // input -> vocabulary spelling
    std::vector<tokenizer_detail::TextStep> decodeSteps_;      // one piece -> text
    std::array<int, 256> byteFallbackIds_{};   // byte -> "<0xNN>" id, -1 when absent
    std::unique_ptr<tokenizer_detail::PieceCodec> codec_ =
        std::make_unique<tokenizer_detail::ByteLevelCodec>();
    bool ignoreMerges_ = false;
    bool byteFallback_ = false;
    bool fuseUnk_ = false;
    std::string unkName_;                      // resolved once the vocabulary is in
    int unkId_ = -1;

    /////////////////////////////////////////////////////////////////////////////////////
    // Encoding
    /////////////////////////////////////////////////////////////////////////////////////

    // Longest added-token whose text starts at `pos`; 0 when none does.
    size_t matchAddedToken(std::string_view text, size_t pos, int& id) const
    {
        int node = addedRoot_[(unsigned char)text[pos]];
        if (node < 0)
            return 0;

        size_t best = 0;
        for (size_t i = pos + 1; ; ++i)
        {
            if (addedTrie_[node].id >= 0) { best = i - pos; id = addedTrie_[node].id; }
            if (i >= text.size())
                break;
            auto next = addedTrie_[node].next.find((unsigned char)text[i]);
            if (next == addedTrie_[node].next.end())
                break;
            node = next->second;
        }
        return best;
    }

    /////////////////////////////////////////////////////////////////////////////////////
    // Loading
    /////////////////////////////////////////////////////////////////////////////////////
    void load(const std::string& path)
    {
        using namespace tinyjson;

        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("tokenizer: cannot open " + path);
        std::vector<char> buf((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (buf.empty())
            throw std::runtime_error("tokenizer: empty file " + path);

        sj_Reader r = sj_reader(buf.data(), buf.size());
        sj_Value root = sj_read(&r);
        if (root.type != SJ_OBJECT)
            throw std::runtime_error("tokenizer: root of " + path + " is not an object");

        sj_Value key, val;
        while (sj_iter_object(&r, root, &key, &val))
        {
            std::string k(key.start, key.end);
            if      (k == "model")          loadModel(r, val, path);
            else if (k == "added_tokens")   loadAddedTokens(r, val);
            else if (k == "pre_tokenizer")  loadPreTokenizer(r, val, path);
            else if (k == "post_processor") loadPostProcessor(r, val);
            else if (k == "normalizer")     loadNormalizer(r, val, path);
            else if (k == "decoder")        loadDecoder(r, val, path);
        }
        if (r.error)
            throw std::runtime_error(std::string("tokenizer: JSON parse error: ") + r.error);
        if (bpeModel_.tokenCount() == 0)
            throw std::runtime_error("tokenizer: no vocabulary found in " + path);
        if (splits_.empty() && codec_->requiresPreTokenizer())
            throw std::runtime_error("tokenizer: no pre-tokenizer split pattern found in " + path);
        // Fusing consecutive unknown tokens only changes anything if an unknown
        // token can be emitted at all, which byte fallback rules out.
        if (fuseUnk_ && !byteFallback_)
            throw std::runtime_error("tokenizer: model.fuse_unk is not supported — " + path);

        codec_->validate(bpeModel_, path);
        buildByteFallbackTable(path);
        buildAddedTokenTrie();

        for (const auto& name : prefixNames_)
            if (int id = pieceToId(name); id >= 0) prefixIds_.push_back(id);
        for (const auto& name : suffixNames_)
            if (int id = pieceToId(name); id >= 0) suffixIds_.push_back(id);
        unkId_ = unkName_.empty() ? -1 : pieceToId(unkName_);
    }

    void buildAddedTokenTrie()
    {
        addedRoot_.fill(-1);
        addedTrie_.clear();

        for (const auto& [content, id] : addedByContent_)
        {
            if (content.empty())
                continue;

            int node = addedRoot_[(unsigned char)content[0]];
            if (node < 0)
            {
                node = int(addedTrie_.size());
                addedTrie_.emplace_back();
                addedRoot_[(unsigned char)content[0]] = node;
            }

            for (size_t i = 1; i < content.size(); ++i)
            {
                unsigned char c = (unsigned char)content[i];
                auto next = addedTrie_[node].next.find(c);
                if (next != addedTrie_[node].next.end()) { node = next->second; continue; }

                int child = int(addedTrie_.size());
                addedTrie_.emplace_back();               // may reallocate, so re-index below
                addedTrie_[node].next.emplace(c, child);
                node = child;
            }
            addedTrie_[node].id = id;
        }
    }

    // Every byte of a SentencePiece vocabulary has to be sayable, or a text the
    // merges cannot cover would silently lose characters.
    void buildByteFallbackTable(const std::string& path)
    {
        byteFallbackIds_.fill(-1);
        if (!byteFallback_)
            return;

        bpeModel_.forEachToken([&](std::string_view piece, int id) {
            int byte = tokenizer_detail::byteFallbackValue(piece);
            if (byte >= 0)
                byteFallbackIds_[byte] = id;
            else if (piece.size() == 1 && byteFallbackIds_[(unsigned char)piece[0]] < 0)
                byteFallbackIds_[(unsigned char)piece[0]] = id;
        });
        for (int b = 0; b < 256; ++b)
            if (byteFallbackIds_[b] < 0)
                throw std::runtime_error(
                    "tokenizer: model.byte_fallback is set but the vocabulary has no token for "
                    "byte " + std::to_string(b) + " — " + path);
    }

    // Steps this class does not implement are refused rather than skipped: they
    // would change the result silently.
    void loadNormalizer(tinyjson::sj_Reader& r, tinyjson::sj_Value node, const std::string& path)
    {
        using namespace tinyjson;
        if (node.type == SJ_NULL)
            return;
        if (node.type != SJ_OBJECT)
            throw std::runtime_error("tokenizer: normalizer is not an object — " + path);

        std::string type;
        tokenizer_detail::TextStep step;
        bool haveFrom = false, haveTo = false;
        sj_Value k, v;
        while (sj_iter_object(&r, node, &k, &v))
        {
            std::string key(k.start, k.end);
            if (key == "type")
            {
                type = tokenizer_detail::jsonString(v);
            }
            else if (key == "normalizers")
            {
                sj_Value child;
                while (sj_iter_array(&r, v, &child))
                    loadNormalizer(r, child, path);
            }
            else if (key == "prepend")
            {
                step.to = tokenizer_detail::jsonString(v);
                haveTo = true;
            }
            else if (key == "content")
            {
                step.to = tokenizer_detail::jsonString(v);
                haveTo = true;
            }
            else if (key == "pattern" && v.type == SJ_OBJECT)
            {
                sj_Value pk, pv;
                while (sj_iter_object(&r, v, &pk, &pv))
                {
                    if (std::string(pk.start, pk.end) != "String")
                        throw std::runtime_error(
                            "tokenizer: only a String pattern is supported in the normalizer — " + path);
                    step.from = tokenizer_detail::jsonString(pv);
                    haveFrom = true;
                }
            }
            else
            {
                throw std::runtime_error(
                    "tokenizer: normalizer field \"" + key + "\" is not supported — " + path);
            }
        }

        if (type == "Sequence")
            return;   // the children registered themselves
        if (type == "Prepend" && haveTo)
            step.kind = tokenizer_detail::TextStep::prepend;
        else if (type == "Replace" && haveFrom && haveTo)
            step.kind = tokenizer_detail::TextStep::replace;
        else
            throw std::runtime_error("tokenizer: normalizer \"" + type + "\" is not supported — " + path);

        normalizeSteps_.push_back(std::move(step));
    }

    // The decoder also tells the two vocabulary encodings apart: a ByteFallback
    // step means pieces are raw SentencePiece text.
    void loadDecoder(tinyjson::sj_Reader& r, tinyjson::sj_Value node, const std::string& path)
    {
        using namespace tinyjson;
        if (node.type != SJ_OBJECT)
            throw std::runtime_error("tokenizer: decoder is not an object — " + path);

        std::string type;
        tokenizer_detail::TextStep step;
        bool haveFrom = false, haveTo = false;
        sj_Value k, v;
        while (sj_iter_object(&r, node, &k, &v))
        {
            std::string key(k.start, k.end);
            if (key == "type")
            {
                type = tokenizer_detail::jsonString(v);
            }
            else if (key == "decoders")
            {
                sj_Value child;
                while (sj_iter_array(&r, v, &child))
                    loadDecoder(r, child, path);
            }
            else if (key == "content")
            {
                step.to = tokenizer_detail::jsonString(v);
                haveTo = true;
            }
            else if (key == "pattern" && v.type == SJ_OBJECT)
            {
                sj_Value pk, pv;
                while (sj_iter_object(&r, v, &pk, &pv))
                {
                    if (std::string(pk.start, pk.end) != "String")
                        throw std::runtime_error(
                            "tokenizer: only a String pattern is supported in the decoder — " + path);
                    step.from = tokenizer_detail::jsonString(pv);
                    haveFrom = true;
                }
            }
            else if (key == "add_prefix_space" || key == "trim_offsets" || key == "use_regex")
            {
                // A decoder ByteLevel carries the pre-tokenizer's settings for
                // symmetry. Decoding maps characters back through the byte table
                // and reports no offsets, so none of the three reaches the result.
            }
            else
            {
                throw std::runtime_error(
                    "tokenizer: decoder field \"" + key + "\" is not supported — " + path);
            }
        }

        if (type == "Sequence" || type == "ByteLevel")
            return;   // ByteLevel: pieces decode through the byte alphabet instead
        if (type == "ByteFallback")
        {
            codec_ = std::make_unique<tokenizer_detail::SentencePieceCodec>();
            return;
        }
        if (type == "Fuse")
            return;   // pieces are concatenated either way
        if (type == "Strip")
            // Undoes the Prepend normalizer's marker, which sits at the start of
            // an encoded fragment only. decode() is called per token while
            // generation streams and carries no such context, so applying it
            // would eat the space in front of every word instead.
            return;
        if (type == "Replace" && haveFrom && haveTo)
        {
            step.kind = tokenizer_detail::TextStep::replace;
            decodeSteps_.push_back(std::move(step));
            return;
        }
        throw std::runtime_error("tokenizer: decoder \"" + type + "\" is not supported — " + path);
    }

    void loadModel(tinyjson::sj_Reader& r, tinyjson::sj_Value model, const std::string& path)
    {
        using namespace tinyjson;
        sj_Value k, v;
        while (sj_iter_object(&r, model, &k, &v))
        {
            std::string key(k.start, k.end);
            if (key == "type")
            {
                std::string type = tokenizer_detail::jsonString(v);
                if (type != "BPE")
                    throw std::runtime_error("tokenizer: only the BPE model is supported, got " + type);
            }
            else if (key == "ignore_merges")
            {
                ignoreMerges_ = tokenizer_detail::jsonBool(v);
            }
            else if (key == "byte_fallback")
            {
                byteFallback_ = tokenizer_detail::jsonBool(v);
            }
            else if (key == "unk_token" && v.type == SJ_STRING)
            {
                unkName_ = tokenizer_detail::jsonString(v);
            }
            else if (key == "unk_token")
            {
                // null: the model declares no unknown token.
            }
            else if (key == "fuse_unk")
            {
                // Checked once byte_fallback is known, which may be parsed later.
                fuseUnk_ = tokenizer_detail::jsonBool(v);
            }
            else if (key == "dropout")
            {
                // Randomly skips merges, so a non-null value tokenizes differently
                // on every call and cannot be reproduced here.
                if (v.type != SJ_NULL)
                    throw std::runtime_error("tokenizer: model.dropout is not supported — " + path);
            }
            else if (key == "continuing_subword_prefix" || key == "end_of_word_suffix")
            {
                // Both reshape how merges apply; neither is handled here.
                if (v.type == SJ_STRING && v.start != v.end)
                    throw std::runtime_error("tokenizer: model." + key + " is not supported");
            }
            else if (key == "vocab")
            {
                sj_Value pk, pv;
                while (sj_iter_object(&r, v, &pk, &pv))
                {
                    std::string piece = tokenizer_detail::jsonString(pk);
                    int id = int(tokenizer_detail::jsonInt(pv));
                    auto symbol = bpeModel_.registerToken(piece, id);
                    if (id >= 0)
                    {
                        if (size_t(id) >= idToSymbol_.size())
                            idToSymbol_.resize(size_t(id) + 1,
                                               tokenizer_detail::BpeModel::invalidSymbol);
                        idToSymbol_[size_t(id)] = symbol;
                    }
                }
            }
            else if (key == "merges")
            {
                sj_Value entry;
                int rank = 0;
                while (sj_iter_array(&r, v, &entry))
                {
                    std::string left, right;
                    if (entry.type == SJ_ARRAY)
                    {
                        sj_Value part;
                        if (!sj_iter_array(&r, entry, &part)) continue;
                        left = tokenizer_detail::jsonString(part);
                        if (!sj_iter_array(&r, entry, &part)) continue;
                        right = tokenizer_detail::jsonString(part);
                    }
                    else if (entry.type == SJ_STRING)
                    {
                        // Legacy form: the pair is one space-separated string.
                        std::string joined = tokenizer_detail::jsonString(entry);
                        size_t sp = joined.find(' ');
                        if (sp == std::string::npos) continue;
                        left = joined.substr(0, sp);
                        right = joined.substr(sp + 1);
                    }
                    else continue;

                    bpeModel_.addMerge(left, right, rank++);
                }
            }
            else
            {
                throw std::runtime_error("tokenizer: model." + key + " is not supported — " + path);
            }
        }
    }

    void loadAddedTokens(tinyjson::sj_Reader& r, tinyjson::sj_Value arr)
    {
        using namespace tinyjson;
        sj_Value entry;
        while (sj_iter_array(&r, arr, &entry))
        {
            if (entry.type != SJ_OBJECT) continue;
            int id = -1;
            AddedToken tok;
            sj_Value k, v;
            while (sj_iter_object(&r, entry, &k, &v))
            {
                std::string key(k.start, k.end);
                if      (key == "id")      id = int(tokenizer_detail::jsonInt(v));
                else if (key == "content") tok.content = tokenizer_detail::jsonString(v);
                else if (key == "special") tok.special = tokenizer_detail::jsonBool(v);
            }
            if (id < 0 || tok.content.empty()) continue;
            addedByContent_[tok.content] = id;
            addedById_[id] = std::move(tok);
        }
    }

    // Walks the (possibly nested) pre-tokenizer description and collects every
    // Split pattern in the order they are applied.
    void loadPreTokenizer(tinyjson::sj_Reader& r, tinyjson::sj_Value node, const std::string& path)
    {
        using namespace tinyjson;
        if (node.type != SJ_OBJECT)
            return;

        std::string type, pattern, behavior;
        bool useRegex = true;   // ByteLevel splits on its own unless told not to
        bool invert = false;
        sj_Value k, v;
        while (sj_iter_object(&r, node, &k, &v))
        {
            std::string key(k.start, k.end);
            if (key == "type")
            {
                type = tokenizer_detail::jsonString(v);
            }
            else if (key == "use_regex")
            {
                useRegex = tokenizer_detail::jsonBool(v);
            }
            else if (key == "behavior")
            {
                behavior = tokenizer_detail::jsonString(v);
            }
            else if (key == "invert")
            {
                invert = tokenizer_detail::jsonBool(v);
            }
            else if (key == "pretokenizers")
            {
                sj_Value child;
                while (sj_iter_array(&r, v, &child))
                    loadPreTokenizer(r, child, path);
            }
            else if (key == "pattern" && v.type == SJ_OBJECT)
            {
                sj_Value pk, pv;
                while (sj_iter_object(&r, v, &pk, &pv))
                {
                    std::string pkey(pk.start, pk.end);
                    if (pkey == "Regex")
                        pattern = tokenizer_detail::jsonString(pv);
                    else if (pkey == "String")
                        pattern = escapeLiteral(tokenizer_detail::jsonString(pv));
                    else
                        throw std::runtime_error(
                            "tokenizer: pre-tokenizer pattern \"" + pkey + "\" is not supported — " + path);
                }
            }
            else if (key == "add_prefix_space")
            {
                // Prepends a space to the input before splitting, which shifts
                // every token that follows.
                if (tokenizer_detail::jsonBool(v))
                    throw std::runtime_error(
                        "tokenizer: pre_tokenizer.add_prefix_space is not supported — " + path);
            }
            else if (key == "trim_offsets")
            {
                // Only moves the reported offsets, which encode() never produces.
            }
            else
            {
                throw std::runtime_error(
                    "tokenizer: pre-tokenizer field \"" + key + "\" is not supported — " + path);
            }
        }

        if (pattern.empty() && type == "ByteLevel" && useRegex)
            pattern = tokenizer_detail::kByteLevelSplit;
        if (pattern.empty())
            return;   // a Sequence wrapper, or a ByteLevel that leaves splitting alone

        using Split = tokenizer_detail::SplitStep;
        tokenizer_detail::SplitStep step;
        if      (behavior.empty() || behavior == "Isolated") step.behavior = Split::isolated;
        else if (behavior == "MergedWithPrevious")           step.behavior = Split::mergedWithPrevious;
        else if (behavior == "MergedWithNext")               step.behavior = Split::mergedWithNext;
        else if (behavior == "Removed")                      step.behavior = Split::removed;
        else
            throw std::runtime_error(
                "tokenizer: split behavior \"" + behavior + "\" is not supported — " + path);
        if (invert)
            throw std::runtime_error("tokenizer: an inverted Split pattern is not supported — " + path);

        step.pattern = std::make_unique<tokenizer_detail::Regex>(pattern);
        splits_.push_back(std::move(step));
    }

    // The post-processor's fixed tokens: everything a TemplateProcessing "single"
    // sequence places before and after the input.
    void loadPostProcessor(tinyjson::sj_Reader& r, tinyjson::sj_Value node)
    {
        using namespace tinyjson;
        if (node.type != SJ_OBJECT)
            return;

        sj_Value k, v;
        while (sj_iter_object(&r, node, &k, &v))
        {
            std::string key(k.start, k.end);
            if (key == "processors")
            {
                sj_Value child;
                while (sj_iter_array(&r, v, &child))
                    loadPostProcessor(r, child);
            }
            else if (key == "single")
            {
                bool seenSequence = false;
                sj_Value item;
                while (sj_iter_array(&r, v, &item))
                {
                    if (item.type != SJ_OBJECT) continue;
                    sj_Value ik, iv;
                    while (sj_iter_object(&r, item, &ik, &iv))
                    {
                        std::string ikey(ik.start, ik.end);
                        if (ikey == "Sequence") { seenSequence = true; continue; }
                        if (ikey != "SpecialToken") continue;

                        sj_Value sk, sv;
                        while (sj_iter_object(&r, iv, &sk, &sv))
                        {
                            if (std::string(sk.start, sk.end) != "id") continue;
                            // The vocabulary may not be parsed yet, so keep the
                            // name and resolve it after the whole file is read.
                            (seenSequence ? suffixNames_ : prefixNames_)
                                .push_back(tokenizer_detail::jsonString(sv));
                        }
                    }
                }
            }
        }
    }

    static std::string escapeLiteral(const std::string& s)
    {
        static const std::string meta = "\\^$.|?*+()[]{}";
        std::string out;
        for (char c : s)
        {
            if (meta.find(c) != std::string::npos) out += '\\';
            out += c;
        }
        return out;
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// Tokenizer - forwards to Impl
/////////////////////////////////////////////////////////////////////////////////////////
Tokenizer::Tokenizer(const std::string& tokenizerJsonPath)
    : impl_(std::make_unique<Impl>(tokenizerJsonPath))
{
}

Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

std::vector<int> Tokenizer::encode(const std::string& text, bool addSpecial, bool parseSpecial) const
{
    return impl_->encode(text, addSpecial, parseSpecial);
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool skipSpecial) const
{
    return impl_->decode(ids, skipSpecial);
}

int Tokenizer::vocabSize() const { return impl_->vocabSize(); }
int Tokenizer::pieceToId(const std::string& piece) const { return impl_->pieceToId(piece); }
std::string Tokenizer::idToPiece(int id) const { return impl_->idToPiece(id); }
bool Tokenizer::isSpecial(int id) const { return impl_->isSpecial(id); }
const std::vector<int>& Tokenizer::prefixIds() const { return impl_->prefixIds(); }
const std::vector<int>& Tokenizer::suffixIds() const { return impl_->suffixIds(); }

}   // namespace vai::utils
