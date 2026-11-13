/*=============================================================================
  Written by   oneway@vaistrata.com
  Task         Llama 3 — multi-turn interactive chat. 
  Assets       data/tokenizers/llama3/tokenizer.json
                 from  huggingface.co/meta-llama/Llama-3.2-1B-Instruct
                 or  huggingface.co/unsloth/Llama-3.2-1B-Instruct
=============================================================================*/

#include "models/production/llama3/net.h"
#include "models/production/llama3/meta-weights.h"
#include "vai/utils/tokenizer.h"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vai;
namespace llama3 = model::production::llama3;


/////////////////////////////////////////////////////////////////////////////////////////
// Chat
/////////////////////////////////////////////////////////////////////////////////////////
// The chat template is declared by "chat_template" in tokenizer_config.json,
// another file that comes alongside the weights (docs/huggingface-model-files.md).
// It is a Jinja2 script and vAi does not run it, so its pieces are spelled out here.
constexpr const char* kBeginOfText = "<|begin_of_text|>";
constexpr const char* kUserHeader = "<|start_header_id|>user<|end_header_id|>\n\n";
constexpr const char* kAssistantHeader = "<|start_header_id|>assistant<|end_header_id|>\n\n";
constexpr const char* kEndOfTurn = "<|eot_id|>";


// Which ids end a turn is declared by "eos_token_id" in generation_config.json,
// one of the files that come alongside the weights (docs/huggingface-model-files.md).
// vAi loads only the weights and the tokenizer, so the names are listed here.
std::vector<int> endOfTurnIds(const utils::Tokenizer& tokenizer)
{
    std::vector<int> ids;
    for (const char* name : { "<|end_of_text|>", "<|eom_id|>", "<|eot_id|>" })
    {
        int id = tokenizer.pieceToId(name);
        if (id < 0)
            throw std::runtime_error(std::string("tokenizer has no end-of-turn token ") + name);
        ids.push_back(id);
    }
    return ids;
}


struct ChatSession
{
    llama3::Net& net;
    const utils::Tokenizer& tokenizer;
    std::vector<int> eogIds;

    ChatSession(llama3::Net& net, const utils::Tokenizer& tokenizer)
    : net(net)
    , tokenizer(tokenizer)
    , eogIds(endOfTurnIds(tokenizer))
    {
    }

    // Prefills `prompt` on top of whatever the KV cache already holds, then
    // decodes until an end-of-turn token, streaming the reply as it arrives.
    // nullopt means the cache filled up and the conversation cannot continue.
    std::optional<std::string> generate(const std::string& prompt)
    {
        // The rendered delta already starts with BOS on the first turn.
        std::vector<int> tokens = tokenizer.encode(prompt);

        if (net.tokensSoFar_ + tokens.size() > net.runtimeCfg.maxRuntimeContext)
            return std::nullopt;

        Tensor inTokens = Tensor({ (uint32_t)tokens.size() }).setData(tokens.data());
        Tensor outToken;

        net.prefill(inTokens, &outToken);
        int nextToken = outToken.getData<uint32_t>()[0];

        std::string reply;

        while (std::find(eogIds.begin(), eogIds.end(), nextToken) == eogIds.end())
        {
            if (net.tokensSoFar_ + 1 > net.runtimeCfg.maxRuntimeContext)
                return std::nullopt;

            // A character can span several tokens, so its bytes leave in pieces.
            // A terminal reassembles them; another consumer would need buffering.
            std::string piece = tokenizer.decode({ nextToken });
            printf("%s", piece.c_str());
            fflush(stdout);
            reply += piece;

            net.decode(&outToken);
            nextToken = outToken.getData<uint32_t>()[0];
        }

        return reply;
    }
};


/////////////////////////////////////////////////////////////////////////////////////////
// chat - multi-turn session. The conversation lives in the KV cache, so each turn
// only has to hand over the text written since the last one.
/////////////////////////////////////////////////////////////////////////////////////////
void chat(llama3::Net& net, const utils::Tokenizer& tokenizer)
{
    ChatSession session(net, tokenizer);

    printf("\n=== Chat (blank line to quit) ===\n");

    bool firstTurn = true;

    while (true)
    {
        printf("\n> ");
        fflush(stdout);

        std::string userMsg;
        if (!std::getline(std::cin, userMsg) || userMsg.empty())
            break;

        // Generation stops *on* the end-of-turn token without consuming it, so
        // the previous reply is still open in the cache: close it here, so the
        // cache matches the training format.
        std::string delta =
            std::string(firstTurn ? kBeginOfText : kEndOfTurn) + kUserHeader + userMsg + kEndOfTurn + kAssistantHeader;
        firstTurn = false;

        printf("\n");
        std::optional<std::string> reply = session.generate(delta);
        if (!reply)
        {
            printf("\n[context full — %u of %u tokens used]\n", net.tokensSoFar_, net.runtimeCfg.maxRuntimeContext);
            break;
        }
        printf("\n");
    }

    printf("\n--- Chat ended ---\n");
}


int main()
{
    // Load model
    auto modelCfg = llama3::ModelConfig::_3_2_1B();
    printf("Loading Llama-%s model...\n", modelCfg.variant);
    llama3::Net net(
        {
            .wDefaultDtype = Dtype::Fp16,
        },
        modelCfg,
        {
            .maxRuntimeContext = 4096,
        });

    // Load weights
    printf("Loading weights...\n");
    loadWeights<"meta">(net);
    printf("Weights loaded successfully\n");

    // Load tokenizer
    printf("Loading tokenizer...\n");
    utils::Tokenizer tokenizer(VAI_DATA_DIR "/tokenizers/llama3/tokenizer.json");
    printf("Tokenizer loaded (vocab_size: %d)\n", tokenizer.vocabSize());

    chat(net, tokenizer);

    return 0;
}
