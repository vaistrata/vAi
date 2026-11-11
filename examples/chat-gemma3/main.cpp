/*=============================================================================
  Written by   oneway@vaistrata.com
  Task         Gemma 3 4B — multi-turn interactive chat with optional images.
               Use <image:file> in your message; replace file with an image filename.
  Assets       data/tokenizers/gemma3/tokenizer.json
                 from  huggingface.co/google/gemma-3-[1b|4b]-it
                 or  huggingface.co/unsloth/gemma-3-[1b|4b]-it
               data/images/<file> (the filename used in <image:file>)
=============================================================================*/

#include "models/production/gemma3/net.h"
#include "models/production/gemma3/google-weights.h"
#include "models/production/gemma3/processing.h"
#include "vai/utils/tokenizer.h"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vai;
namespace gemma3 = model::production::gemma3;


/////////////////////////////////////////////////////////////////////////////////////////
// Chat
/////////////////////////////////////////////////////////////////////////////////////////
// The chat template is declared by "chat_template" in tokenizer_config.json,
// another file that comes alongside the weights (docs/huggingface-model-files.md).
// It is a Jinja2 script and vAi does not run it, so its pieces are spelled out here.
// Gemma calls the assistant "model", and the newline after a closed turn is part
// of the template rather than of the end-of-turn token.
constexpr const char* kBeginOfText = "<bos>";
constexpr const char* kUserHeader = "<start_of_turn>user\n";
constexpr const char* kAssistantHeader = "<start_of_turn>model\n";
constexpr const char* kEndOfTurn = "<end_of_turn>\n";

// Keep below MAX_CAUSAL_BLOCKS / 2 - 1. More images mean a longer vision dispatch
// and a higher risk of GPU timeout; this example does not recover from timeouts.
constexpr uint32_t kMaxImagesPerTurn = 2;


// Which ids end a turn is declared by "eos_token_id" in generation_config.json,
// one of the files that come alongside the weights (docs/huggingface-model-files.md).
// vAi loads only the weights and the tokenizer, so the names are listed here.
std::vector<int> endOfTurnIds(const utils::Tokenizer& tokenizer)
{
    std::vector<int> ids;
    for (const char* name : {"<eos>", "<end_of_turn>"})
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
    gemma3::Net& net;
    const utils::Tokenizer& tokenizer;
    std::vector<int> eogIds;

    ChatSession(gemma3::Net& net, const utils::Tokenizer& tokenizer)
        : net(net), tokenizer(tokenizer), eogIds(endOfTurnIds(tokenizer))
    {
    }

    // Prefills `prompt` and `images` on top of whatever the KV cache already holds,
    // then decodes until an end-of-turn token, streaming the reply as it arrives.
    // nullopt means the cache filled up and the conversation cannot continue.
    std::optional<std::string> generate(const std::string& prompt, const Tensor& images = {})
    {
        // The rendered delta already starts with BOS on the first turn.
        std::vector<int> tokens =
            tokenizer.encode(gemma3::expandImagePlaceholders(prompt, net.modelCfg.visionCfg->tokensPerImage));

        if (net.tokensSoFar_ + tokens.size() > net.runtimeCfg.maxRuntimeContext)
            return std::nullopt;

        Tensor inTokens = Tensor({(uint32_t)tokens.size()}).setData(tokens.data());
        Tensor outToken;

        net.prefill(images, inTokens, &outToken);
        int nextToken = outToken.getData<uint32_t>()[0];

        std::string reply;

        while (std::find(eogIds.begin(), eogIds.end(), nextToken) == eogIds.end())
        {
            if (net.tokensSoFar_ + 1 > net.runtimeCfg.maxRuntimeContext)
                return std::nullopt;

            // A character can span several tokens, so its bytes leave in pieces.
            // A terminal reassembles them; another consumer would need buffering.
            std::string piece = tokenizer.decode({nextToken});
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
void chat(gemma3::Net& net, const utils::Tokenizer& tokenizer)
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

        std::vector<std::string> imagePaths;
        const std::string message = gemma3::takeImageTags(userMsg, imagePaths);
        if (imagePaths.size() > kMaxImagesPerTurn)
        {
            printf("\n[at most %u images per turn]\n", kMaxImagesPerTurn);
            continue;
        }

        Tensor images;
        try
        {
            if (!imagePaths.empty())
            {
                const uint32_t B = (uint32_t)imagePaths.size();
                const uint32_t C = net.modelCfg.visionCfg->imageChannels;
                const uint32_t S = net.modelCfg.visionCfg->imageSize;
                for (std::string& path : imagePaths)
                    path = VAI_DATA_DIR "/images/" + path;
                images = Tensor({B, C, S, S}).setFloats(gemma3::loadImages(imagePaths, S));
            }
        }
        catch (const std::exception& e)
        {
            printf("\n[%s]\n", e.what());
            continue;
        }

        // Generation stops *on* the end-of-turn token without consuming it, so
        // the previous reply is still open in the cache: close it here, so the
        // cache matches the training format.
        std::string delta =
            std::string(firstTurn ? kBeginOfText : kEndOfTurn) + kUserHeader + message + kEndOfTurn + kAssistantHeader;
        firstTurn = false;

        printf("\n");
        std::optional<std::string> reply = session.generate(delta, images);
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
    auto modelCfg = gemma3::ModelConfig::_4B();
    printf("Loading Gemma-%s model...\n", modelCfg.variant);
    gemma3::Net net(
        {
            .wDefaultDtype = Dtype::Fp16,
        },
        modelCfg,
        {
            .maxRuntimeContext = 4096,
        });

    // Load weights
    printf("Loading weights...\n");
    loadWeights<"google">(net);
    printf("Weights loaded successfully\n");

    // Load tokenizer
    printf("Loading tokenizer...\n");
    utils::Tokenizer tokenizer(VAI_DATA_DIR "/tokenizers/gemma3/tokenizer.json");
    printf("Tokenizer loaded (vocab_size: %d)\n", tokenizer.vocabSize());

    chat(net, tokenizer);

    return 0;
}
