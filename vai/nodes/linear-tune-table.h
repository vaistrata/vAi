#ifndef VAI_NODES_LINEAR_TUNE_TABLE_H
#define VAI_NODES_LINEAR_TUNE_TABLE_H

// Measured Linear configurations per shape: exact lookup, else nearest
// neighbours revised to the shape and reranked by the vendor's score.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vai::node {


/*=============================================================================
  LinearTuneTable - the fastest configuration measured offline for a set of
  Linear shapes M x K x N, and linearTunePick(), which turns it into a pick
  for any shape:

    1. exact     the shape is in the table: its configuration, as measured
       (a miss goes to step 2 under LinearTuneMiss::Neighbour, else to 5)
    2. retrieve  the k entries nearest to the shape (k-NN search)
    3. revise    each neighbour's configuration adapted to the shape, or
                 dropped when it cannot run it well (the vendor's revise)
    4. rerank    the vendor's score model ranks what survives; ties go to
                 the nearer neighbour
    5. fallback  nothing survives, the table is empty, or the miss policy is
                 Fallback: the vendor's rule

  A vendor path supplies its configuration type Cfg, the table rows (a
  generated .inc of LinearTuneEntry<Cfg>), revise, score and fallback; the
  mechanism here knows nothing about any kernel. Xe3's xe3LinearChoose()
  (linear.h) is the reference use.

  Distance: Euclidean in (log2 M, log2 K, log2 N). Shapes span 1 .. 1e5+ per
  axis and kernels react to ratios, not differences: M 1 -> 50 is as far as
  N 2048 -> 102400.

  Cost: a brute-force scan, a few hundred 3-D distances, well under a
  microsecond; prepare() runs it once per shape. A spatial index pays off
  only with orders of magnitude more rows.
=============================================================================*/

struct LinearShape
{
    uint32_t M = 0, K = 0, N = 0;

    bool operator==(const LinearShape&) const = default;
};

template <class Cfg>
struct LinearTuneEntry
{
    LinearShape shape;
    Cfg cfg;
};

// Where linearTunePick()'s configuration came from.
enum class LinearTuneSource {
    Exact,       // the table's row for this shape
    Neighbour,   // a nearby row's configuration, revised and reranked
    Fallback,    // the vendor's rule
};

// What linearTunePick() does with a shape the table does not have.
enum class LinearTuneMiss {
    Fallback,    // the vendor's rule
    Neighbour,   // the nearest rows' configurations, revised and reranked
};

// The process's miss policy: VAI_LINEAR_TUNE_MISS=neighbour selects Neighbour,
// anything else (or unset) Fallback. Read once.
inline LinearTuneMiss linearTuneMiss()
{
    static const LinearTuneMiss miss = [] {
        const char* v = std::getenv("VAI_LINEAR_TUNE_MISS");
        return (v && std::strcmp(v, "neighbour") == 0) ? LinearTuneMiss::Neighbour : LinearTuneMiss::Fallback;
    }();
    return miss;
}

template <class Cfg>
struct LinearTuneResult
{
    Cfg cfg;
    LinearTuneSource source;
};

template <class Cfg>
class LinearTuneTable
{
public:
    using Entry = LinearTuneEntry<Cfg>;

    // The rows are referenced, not copied: give it storage that outlives the
    // table (the generated rows are a static array). Shapes must be unique.
    explicit LinearTuneTable(std::span<const Entry> entries)
    : entries_(entries)
    {
        exact_.reserve(entries.size());
        coords_.reserve(entries.size());
        for (uint32_t i = 0; i < entries.size(); ++i)
        {
            const bool unique = exact_.emplace(entries[i].shape, i).second;
            (void)unique;   // a duplicate shape keeps the first row
            coords_.push_back(coords(entries[i].shape));
        }
    }

    size_t size() const { return entries_.size(); }

    // The row for exactly this shape, or nullptr.
    const Entry* exact(const LinearShape& s) const
    {
        const auto it = exact_.find(s);
        return it == exact_.end() ? nullptr : &entries_[it->second];
    }

    // Up to k rows nearest to s, nearest first (equal distances in table order).
    std::vector<const Entry*> nearest(const LinearShape& s, uint32_t k) const
    {
        const Coords q = coords(s);
        std::vector<std::pair<double, uint32_t>> byDistance;   // (squared distance, row)
        byDistance.reserve(entries_.size());
        for (uint32_t i = 0; i < entries_.size(); ++i)
        {
            const Coords& c = coords_[i];
            const double dm = c.m - q.m, dk = c.k - q.k, dn = c.n - q.n;
            byDistance.emplace_back(dm * dm + dk * dk + dn * dn, i);
        }
        const size_t n = std::min<size_t>(k, byDistance.size());
        std::partial_sort(byDistance.begin(), byDistance.begin() + n, byDistance.end());

        std::vector<const Entry*> out;
        out.reserve(n);
        for (size_t i = 0; i < n; ++i)
            out.push_back(&entries_[byDistance[i].second]);
        return out;
    }

private:
    struct Coords
    {
        double m, k, n;
    };
    static Coords coords(const LinearShape& s)
    {
        // max(1, ...): a zero extent would be -inf; it never reaches a real pick.
        auto l = [](uint32_t x) { return std::log2(double(std::max(x, 1u))); };
        return { l(s.M), l(s.K), l(s.N) };
    }
    struct ShapeHash
    {
        size_t operator()(const LinearShape& s) const
        {
            const uint64_t mk = (uint64_t(s.M) << 32) | s.K;
            return std::hash<uint64_t>{}(mk) ^ (std::hash<uint32_t>{}(s.N) * 0x9e3779b97f4a7c15ull);
        }
    };

    std::span<const Entry> entries_;
    std::unordered_map<LinearShape, uint32_t, ShapeHash> exact_;   // shape -> row
    std::vector<Coords> coords_;                                   // per row, log2 space
};

// The pick for shape s (steps 1 - 5 above); miss: what a shape outside the
// table gets (linearTuneMiss() for the process's choice).
//   revise(s, cfg)  -> std::optional<Cfg>  a neighbour's cfg made to run s, or none
//   score(s, cfg)   -> double              higher is better
//   fallback(s)     -> Cfg
template <class Cfg, class Revise, class Score, class Fallback>
LinearTuneResult<Cfg> linearTunePick(
    const LinearTuneTable<Cfg>& table, const LinearShape& s, LinearTuneMiss miss, uint32_t k, Revise&& revise,
    Score&& score, Fallback&& fallback)
{
    if (const auto* hit = table.exact(s))
        return { hit->cfg, LinearTuneSource::Exact };
    if (miss == LinearTuneMiss::Fallback)
        return { fallback(s), LinearTuneSource::Fallback };

    std::optional<Cfg> best;
    double bestScore = 0.0;
    for (const auto* e : table.nearest(s, k))
    {
        const std::optional<Cfg> c = revise(s, e->cfg);
        if (!c)
            continue;
        const double v = score(s, *c);
        if (!best || v > bestScore)   // strict: a tie keeps the nearer neighbour
        {
            best = c;
            bestScore = v;
        }
    }
    if (best)
        return { *best, LinearTuneSource::Neighbour };
    return { fallback(s), LinearTuneSource::Fallback };
}


}   // namespace vai::node

#endif   // VAI_NODES_LINEAR_TUNE_TABLE_H
