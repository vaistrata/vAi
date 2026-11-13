/*=============================================================================
  Written by   oneway@vaistrata.com
  Task         ACT — turns three camera views and the robot's state into one
               action chunk, profiles the forward.
  Assets       data/images/cam_high.jpg, cam_right_wrist.jpg, cam_left_wrist.jpg
                 from  huggingface.co/datasets/suz22/RoboTwin_clean
=============================================================================*/

#include "net.h"
#include "lerobot-weights.h"
#include "vai/utils/image-processor.h"
#include "vai/utils/time-checker.h"
#include <algorithm>
#include <iterator>
#include <cstdio>
#include <string>
#include <vector>

using namespace vai;
namespace act = model::production::act;


/////////////////////////////////////////////////////////////////////////////////////////
// Tag nodes for profiling — group by kernel shape, so identical-shape ops across
// a tower collapse to one row per distinct kernel.
/////////////////////////////////////////////////////////////////////////////////////////
void setProfileTags(act::Net& net)
{
    auto lin = [](const char* tower, uint32_t in, uint32_t out) {
        return std::string(tower) + ".Linear.[" + std::to_string(in) + "x" + std::to_string(out) + "]";
    };

    // Vision tower — ResNet-18, a conv per kernel, stride and Ci x Co.
    auto tagConv = [](vai::node::Convolution& conv) {
        const auto& c = conv.config_;
        conv.setProfileTag("Vision.Conv" + std::to_string(c.kernelSize) + "x" + std::to_string(c.kernelSize) + "s" +
                           std::to_string(c.stride) + ".[" + std::to_string(c.inChannels) + "x" +
                           std::to_string(c.outChannels) + "]");
    };

    const auto& modelCfg = net.modelCfg;
    const uint32_t D = modelCfg.embedDim;         // 512
    const uint32_t Dff = modelCfg.ffnHiddenDim;   // 3200

    tagConv(net.imageEncoder.conv1.conv);
    net.imageEncoder.maxPool.setProfileTag("Vision.MaxPool");
    for (auto& b : net.imageEncoder.blocks)
    {
        tagConv(b.conv1.conv);
        tagConv(b.conv2.conv);
        if (b.shortcut)
            tagConv(b.shortcut->conv);
    }
    net.transpose.setProfileTag("Vision.Permute");
    net.imageProj.setProfileTag(lin("Vision", vai::group::ResNet18::outChannels, D));

    // Encoder — one shape for every block.
    net.latentStateProj.setProfileTag(lin("Encoder", modelCfg.stateDim, 2 * D));
    net.catTokens.setProfileTag("Encoder.Concat");
    for (auto& b : net.encoder)
    {
        b.mhAttn.projQKV.setProfileTag(lin("Encoder", D, 3 * D));
        b.mhAttn.perQKV.setProfileTag("Encoder.Permute");
        b.mhAttn.attn.setProfileTag("Encoder.Attn");
        b.mhAttn.projOut.setProfileTag(lin("Encoder", D, D));
        b.norm1.setProfileTag("Encoder.LayerNorm");
        b.ffn.fc1.setProfileTag(lin("Encoder", D, Dff));
        b.ffn.fc2.setProfileTag(lin("Encoder", Dff, D));
        b.norm2.setProfileTag("Encoder.LayerNorm");
    }

    // Decoder — the cross-attention over the encoder's output, then the action head.
    auto& dec = net.decoder;
    dec.projKV.setProfileTag(lin("Decoder", D, 2 * D));
    dec.perKV.setProfileTag("Decoder.Permute");
    dec.attn.setProfileTag("Decoder.Attn");
    dec.projOut.setProfileTag(lin("Decoder", D, D));
    dec.norm1.setProfileTag("Decoder.LayerNorm");
    dec.ffn.fc1.setProfileTag(lin("Decoder", D, Dff));
    dec.ffn.fc2.setProfileTag(lin("Decoder", Dff, D));
    dec.norm2.setProfileTag("Decoder.LayerNorm");
    net.decoderNorm.setProfileTag("Decoder.LayerNorm");
    net.actionHead.setProfileTag(lin("Decoder", D, modelCfg.actionDim));
}


int main()
{
    // Construct model — mair-robosyn/ACT_sim_item_assembly_official, as its config.json sets it.
    auto modelCfg = act::ModelConfig::get({
        .variant = "mair-robosyn/item-assembly",
        .cameras = { "cam_high", "cam_right_wrist", "cam_left_wrist" },
        .imageSize = { 480, 640 },
        .stateDim = 14,
        .actionDim = 14,
        .actionChunk = 50,
    });
    printf("Loading ACT %s model...\n", modelCfg.variant);
    act::Net net(
        {
            .wDefaultDtype = Dtype::Fp16,
        },
        modelCfg);

    // Configure the per-node GPU execution-time profiler
    setProfileTags(net);

    // Load weights
    printf("Loading weights...\n");
    {
        utils::TimeChecker loadTimer("Weights loaded successfully in {}ms");
        loadWeights<"lerobot">(net);
    }

    const std::pair<std::string, const char*> cameraImages[] = {
        { modelCfg.cameras[0], VAI_DATA_DIR "/images/cam_high.jpg" },
        { modelCfg.cameras[1], VAI_DATA_DIR "/images/cam_right_wrist.jpg" },
        { modelCfg.cameras[2], VAI_DATA_DIR "/images/cam_left_wrist.jpg" },
    };
    const uint32_t B = (uint32_t)std::size(cameraImages);
    const auto [H, W] = modelCfg.imageSize;
    VAI_ASSERT(B == modelCfg.numCameras());

    // Preprocess the images, each with its camera's statistics, and stack them into [B, 3, H, W]
    std::vector<float> pixels;
    pixels.reserve(size_t(B) * 3 * H * W);
    for (const auto& [name, path] : cameraImages)
    {
        printf("Loading image %s: %s\n", name.c_str(), path);
        std::vector<float> chw = utils::ImageProcessor(path)
                                     .normalize(0.0f, 1.0f)
                                     .standardize(net.imageMean.at(name), net.imageStd.at(name))
                                     .chw();
        VAI_ASSERT(chw.size() == size_t(3) * H * W);
        pixels.insert(pixels.end(), chw.begin(), chw.end());
    }

    // The state is the robot's joint positions; the arms at rest.
    std::vector<float> states(modelCfg.stateDim, 0.0f);
    net.normalizeState(states);

    // Initialize the input tensors
    Tensor inImages = Tensor({ B, 3, H, W }).setFloats(std::move(pixels));
    Tensor inState = Tensor({ 1, modelCfg.stateDim }).setFloats(std::move(states));
    Tensor actions;

    // Warm-up.
    {
        net.predict(inImages, inState, &actions);
    }

    net.resetProfileRecords();

    // Forward — every camera and the state in one pass, to the whole chunk.
    {
        utils::TimeChecker totalTimer("\n"
                                      "────────────────────────────────────────\n"
                                      "▶ Forward {} images + state -> {} actions in {}ms \n"
                                      "────────────────────────────────────────",
                                      B,
                                      modelCfg.actionChunk);

        net.predict(inImages, inState, &actions);

        net.sync();   // this sync is not necessary, but it is here to measure the forward time.
    }

    net.printProfile();

    // Print the chunk — the first few steps and dims, in joint units.
    printf("\n=== Action chunk (%u steps x %u dims) ===\n", modelCfg.actionChunk, modelCfg.actionDim);
    const uint32_t showSteps = std::min(modelCfg.actionChunk, 5u);
    const uint32_t showDims = std::min(modelCfg.actionDim, 8u);
    const float* data = actions.getData<float>();
    std::vector<float> a(data, data + size_t(modelCfg.actionChunk) * modelCfg.actionDim);
    net.denormalizeActions(a);
    for (uint32_t s = 0; s < showSteps; ++s)
    {
        printf("  t%-2u ", s);
        for (uint32_t d = 0; d < showDims; ++d)
            printf("% .4f ", a[size_t(s) * modelCfg.actionDim + d]);
        printf("...\n");
    }
    printf("  ...\n");

    printf("\n--- Action chunk complete ---\n");

    return 0;
}


/*
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! Don't erase the following reference output, it is used to verify the model's correctness.  !!
  !!                                                                                            !!
  !! Reference chunk [50 steps x 14 dims], in joint units: the run above with every dtype       !!
  !! fp32 (weights, activations) and the naive attention kernel.                                !!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  t0  -0.35501593  0.59358412 -0.39993280  0.26309597  0.20642757  0.33939660  0.29335767  0.05797081
       1.15864921 -0.54498601 -0.06495953 -0.17657256  0.65771568  0.29427966
  t1  -0.36409318  0.59996730 -0.40459165  0.25780112  0.19587332  0.36196333  0.28023264  0.06252731
       1.17222238 -0.55443478 -0.06269015 -0.18636802  0.67131913  0.28079033
  t2  -0.37402898  0.60245883 -0.40735176  0.26579869  0.19693306  0.36891395  0.26872265  0.06539053
       1.17892087 -0.55704081 -0.06451314 -0.19494516  0.69112283  0.26897043
  t3  -0.37603438  0.61229891 -0.41160429  0.25508192  0.19822830  0.37913638  0.26439124  0.06713831
       1.19472837 -0.56880909 -0.06338675 -0.18861669  0.69084042  0.26463568
  t4  -0.37775016  0.62131858 -0.41523629  0.25044912  0.20447981  0.38372988  0.25795424  0.06821017
       1.21036172 -0.57944804 -0.06075522 -0.17833921  0.69140875  0.25822234
  t5  -0.38379833  0.61633736 -0.41546330  0.25131312  0.19657353  0.39936489  0.25339729  0.07035097
       1.20774508 -0.57915336 -0.06139845 -0.19117451  0.70752269  0.25342226
  t6  -0.38530713  0.61670619 -0.41635308  0.24930349  0.19389692  0.40442789  0.24580957  0.07140398
       1.21123099 -0.58196437 -0.06130203 -0.19571805  0.71159041  0.24572046
  t7  -0.38848257  0.62061870 -0.41864160  0.24646860  0.19462717  0.41407561  0.24385223  0.07251732
       1.21938729 -0.58832449 -0.06097423 -0.19515100  0.71963298  0.24366570
  t8  -0.39229324  0.62666458 -0.42041701  0.24805194  0.20261022  0.41406447  0.23748915  0.07366757
       1.22904098 -0.59333920 -0.06031569 -0.18712944  0.72333789  0.23735082
  t9  -0.39424649  0.63295096 -0.42395863  0.24404311  0.20454952  0.41821581  0.23434213  0.07464202
       1.24138045 -0.60307705 -0.05876396 -0.18536487  0.72311825  0.23415168
  t10 -0.39571333  0.63719261 -0.42653304  0.23705035  0.20633420  0.43002528  0.22980557  0.07529952
       1.25263190 -0.61243975 -0.05638630 -0.18178290  0.72842574  0.22963443
  t11 -0.39886573  0.63717777 -0.42774311  0.23424053  0.20217931  0.44084537  0.22555119  0.07686520
       1.25593305 -0.61594361 -0.05555531 -0.18724528  0.73583561  0.22525261
  t12 -0.40135494  0.64371818 -0.43119693  0.22944978  0.20431653  0.45386720  0.22222611  0.07847051
       1.27215052 -0.63047367 -0.05199280 -0.18077767  0.74087101  0.22193825
  t13 -0.40711898  0.63978636 -0.43265301  0.23200026  0.19949427  0.46625888  0.21762516  0.07956210
       1.27348518 -0.63306373 -0.05261133 -0.19531506  0.75679195  0.21713513
  t14 -0.40779877  0.64483494 -0.43485415  0.22026840  0.19703820  0.48155969  0.21648221  0.08066538
       1.28529739 -0.64433044 -0.04918585 -0.19215450  0.75756741  0.21599171
  t15 -0.41043681  0.64379412 -0.43639505  0.22235698  0.19682875  0.48571718  0.20959036  0.08159551
       1.28756130 -0.64682597 -0.04878304 -0.19597885  0.76584208  0.20901436
  t16 -0.41669008  0.65193969 -0.44058567  0.22582480  0.20518187  0.49002224  0.20423988  0.08302277
       1.30425870 -0.65918761 -0.04794464 -0.19050065  0.77462351  0.20359156
  t17 -0.41962653  0.64771116 -0.44125256  0.22321570  0.19668165  0.51105195  0.19822329  0.08368225
       1.30368710 -0.66174197 -0.04743969 -0.20360637  0.79065663  0.19742209
  t18 -0.42236501  0.64585036 -0.44209409  0.21584660  0.18604904  0.52872503  0.19554006  0.08489805
       1.30445623 -0.66492331 -0.04647154 -0.21234643  0.79738069  0.19463088
  t19 -0.42570093  0.64623481 -0.44458646  0.21481481  0.18344870  0.53353268  0.19503465  0.08592437
       1.30786896 -0.66995960 -0.04641053 -0.21857819  0.80239570  0.19398689
  t20 -0.42668226  0.65182209 -0.44665265  0.20407268  0.17928487  0.55376750  0.19303459  0.08712135
       1.31811607 -0.67909276 -0.04542804 -0.21525344  0.80895656  0.19189417
  t21 -0.43060145  0.64913499 -0.44710520  0.20583659  0.17629716  0.56170189  0.18986890  0.08701649
       1.31634235 -0.67829573 -0.04673255 -0.22509348  0.81983429  0.18861561
  t22 -0.43576208  0.64852852 -0.45007384  0.21142015  0.17858326  0.56555206  0.18444149  0.08785319
       1.32420480 -0.68623877 -0.04568645 -0.22949651  0.82945603  0.18307519
  t23 -0.43725491  0.64882249 -0.45164835  0.20374998  0.17051989  0.58247709  0.17739865  0.08835472
       1.32266998 -0.68578321 -0.04756271 -0.23990673  0.84038693  0.17580202
  t24 -0.44355017  0.64809459 -0.45484427  0.21263948  0.17232400  0.58778781  0.17008337  0.08872174
       1.32365251 -0.68775421 -0.04938197 -0.24985558  0.85718673  0.16832978
  t25 -0.44489321  0.64715838 -0.45595974  0.20464787  0.16615650  0.60292983  0.16764073  0.08917128
       1.32633853 -0.69161654 -0.04777835 -0.25405738  0.86327028  0.16574121
  t26 -0.44868812  0.64689690 -0.45848781  0.20976791  0.16746783  0.60175920  0.16537017  0.08907335
       1.32837653 -0.69437915 -0.04948294 -0.26178455  0.86979771  0.16334179
  t27 -0.45097178  0.64943206 -0.46089289  0.21320128  0.16751364  0.60666603  0.16280405  0.08932453
       1.33224154 -0.69820040 -0.04904157 -0.26471600  0.87726015  0.16064820
  t28 -0.45349792  0.65093726 -0.46323022  0.20928261  0.16447181  0.61404377  0.16224232  0.09012516
       1.33479893 -0.70190418 -0.04934213 -0.26975825  0.88094395  0.15994404
  t29 -0.45785579  0.65571856 -0.46772486  0.21405765  0.16529879  0.62019211  0.15561447  0.09084615
       1.33996546 -0.70743668 -0.04975600 -0.27578935  0.89308923  0.15317386
  t30 -0.46394825  0.64973068 -0.46757808  0.22389606  0.15913180  0.63011783  0.15339065  0.09202795
       1.33365655 -0.70384967 -0.05159440 -0.29127240  0.91111290  0.15064412
  t31 -0.46641091  0.65524024 -0.47239631  0.22493634  0.15917850  0.63907903  0.14695844  0.09231740
       1.34207511 -0.71256196 -0.05040280 -0.29139069  0.91779846  0.14402755
  t32 -0.47058320  0.65417516 -0.47424757  0.23545191  0.16024309  0.63929033  0.14051214  0.09257129
       1.33857274 -0.71070814 -0.05106784 -0.29945746  0.92885667  0.13745375
  t33 -0.46992812  0.65702462 -0.47684360  0.22762918  0.14803699  0.64931607  0.14016181  0.09315279
       1.33882749 -0.71348387 -0.05161370 -0.30892706  0.92992353  0.13696504
  t34 -0.47199488  0.66002697 -0.48052040  0.22665137  0.14846340  0.65600026  0.13755688  0.09331083
       1.34535539 -0.72030586 -0.05081391 -0.30957809  0.93443656  0.13419789
  t35 -0.47509027  0.66311300 -0.48371005  0.23487899  0.14681315  0.65645480  0.13635132  0.09395321
       1.34491169 -0.72164476 -0.05194305 -0.31621155  0.94267619  0.13288358
  t36 -0.47231725  0.66115606 -0.48373929  0.22181571  0.13527316  0.66683966  0.13851541  0.09316252
       1.33935547 -0.71889174 -0.05217428 -0.32440898  0.93961155  0.13498595
  t37 -0.48026514  0.66276431 -0.48813054  0.24912861  0.14384028  0.65774155  0.13175863  0.09440470
       1.34277678 -0.72204936 -0.05342340 -0.33157960  0.95672810  0.12813550
  t38 -0.47841787  0.65832692 -0.48567700  0.23456648  0.13367876  0.67783231  0.13284867  0.09412636
       1.33631897 -0.71701407 -0.05349658 -0.33739448  0.96058142  0.12913081
  t39 -0.48001188  0.65234512 -0.48672885  0.24857771  0.12748107  0.66667360  0.13233989  0.09361628
       1.32377744 -0.70940745 -0.05683678 -0.35602602  0.96475869  0.12850358
  t40 -0.48316967  0.66424620 -0.49200320  0.25525686  0.13636267  0.66144097  0.12703651  0.09438576
       1.33745158 -0.71958345 -0.05552278 -0.34814975  0.96750462  0.12320027
  t41 -0.48252130  0.66597188 -0.49253175  0.24745148  0.12975490  0.66942668  0.12832822  0.09479250
       1.34115350 -0.72405493 -0.05429818 -0.34803495  0.96493930  0.12450685
  t42 -0.48476225  0.66488445 -0.49388292  0.26349905  0.13766328  0.66039211  0.12578808  0.09437369
       1.33769929 -0.72035128 -0.05564200 -0.35199898  0.97459209  0.12195574
  t43 -0.48601502  0.66660142 -0.49562082  0.25660914  0.12949458  0.67502755  0.12432133  0.09541886
       1.34288502 -0.72665101 -0.05504371 -0.35505912  0.97848904  0.12045196
  t44 -0.48862582  0.66957814 -0.49763709  0.25957075  0.12632740  0.67950737  0.12427768  0.09624574
       1.34271884 -0.72812307 -0.05517927 -0.35858232  0.98396003  0.12035401
  t45 -0.49224579  0.66028225 -0.49761996  0.27474886  0.11632437  0.67746854  0.12164460  0.09608337
       1.32642269 -0.71695471 -0.05900214 -0.38442355  0.99725199  0.11744913
  t46 -0.49155107  0.66519237 -0.50080109  0.27749252  0.11799270  0.67602217  0.11727592  0.09578340
       1.32926083 -0.72003686 -0.05892845 -0.38327917  0.99846852  0.11309512
  t47 -0.49552655  0.66961837 -0.50494945  0.28694066  0.12023264  0.67285717  0.11189154  0.09578227
       1.33230102 -0.72155344 -0.06035162 -0.39080620  1.00738800  0.10758880
  t48 -0.49789730  0.66992068 -0.50689805  0.29846373  0.11929774  0.66855645  0.10784572  0.09626251
       1.32863009 -0.71973908 -0.06138975 -0.39763415  1.01345885  0.10347310
  t49 -0.50357455  0.68544477 -0.51578695  0.32087478  0.14169720  0.65352988  0.09874275  0.09600031
       1.34418166 -0.72900140 -0.06106393 -0.38952398  1.02437055  0.09437160
*/
