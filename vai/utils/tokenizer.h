/*=============================================================================
  Written by   oneway@vaistrata.com
  BPE tokenizer driven entirely by a HuggingFace tokenizer.json.

  Everything that decides tokenization comes out of the file: the vocabulary,
  the merge ranks, the added tokens, the normalizer, the pre-tokenizer's split
  regex, and the post-processor's leading/trailing special tokens. No
  model-specific code, so the same class serves GPT-2, Llama, Qwen, Mistral,
  Gemma and anything else whose tokenizer.json uses a BPE model.

  Two vocabulary encodings are supported, told apart by the decoder the file
  declares:
    ByteLevel      every byte is remapped to a printable code point, so a piece
                   is text and arbitrary bytes still round-trip (GPT-2, Llama3).
    SentencePiece  pieces hold raw text with U+2581 standing in for a space,
                   and bytes the vocabulary cannot spell fall back to <0xNN>
                   tokens (Llama2, Gemma). Marked by a ByteFallback decoder.

  The split regex is interpreted by a small engine covering the constructs those
  patterns use (alternation, classes with \p{L} / \p{N} / \s, greedy quantifiers,
  (?i:) groups, lookahead). Anything outside that subset throws at load time
  rather than silently tokenizing differently from the reference implementation.

  Unicode property classes are matched from a range table that covers the major
  scripts. Rare scripts may classify differently than the reference.

    Tokenizer tok("tokenizer.json");
    auto ids  = tok.encode("Hello", true);   // addSpecial
    auto text = tok.decode(ids);
=============================================================================*/

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <memory>
#include <string>
#include <vector>


class Tokenizer
{
public:
    // Throws if the file is missing, malformed, or uses a construct outside the
    // supported subset. Unicode NFC is one: vAi does not compose. inputIsNfc
    // says the caller hands encode() text that is NFC already (normalized with
    // java.text.Normalizer or ICU, or ASCII), so an NFC normalizer is taken as
    // done rather than refused (Qwen2 / Qwen3).
    explicit Tokenizer(const std::string& tokenizerJsonPath, bool inputIsNfc = false);
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    // addSpecial applies the post-processor's leading/trailing special tokens
    // (BOS and friends); off by default because chat templates already spell
    // those out. parseSpecial recognises added-token strings such as
    // <|eot_id|> in the input and emits them as their own id instead of letting
    // BPE split them apart.
    std::vector<int> encode(const std::string& text, bool addSpecial = false, bool parseSpecial = true) const;

    // Decodes ids to text. A call carries no sequence context, so it is safe to
    // feed one token at a time as generation streams them.
    std::string decode(const std::vector<int>& ids, bool skipSpecial = true) const;

    int vocabSize() const;

    // Returns -1 when the piece is not in the vocabulary.
    int pieceToId(const std::string& piece) const;
    std::string idToPiece(int id) const;
    bool isSpecial(int id) const;

    // Ids the post-processor prepends / appends when encode(addSpecial = true).
    const std::vector<int>& prefixIds() const;
    const std::vector<int>& suffixIds() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // TOKENIZER_H
