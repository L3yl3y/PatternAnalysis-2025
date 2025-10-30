HEYO HEYO :3

Running these command:
    cd recognition/Leyla\ Suljic\ 48912196\ ConvNeXt\ Classify\ Alzheimer’s\ Disease\ Of\ The\ ADNI\ Brain\ Data\ /data
    scp -r s4891219@rangpur.compute.eait.uq.edu.au:/home/groups/comp3710/ADNI .

===================================================================================================================================================================================
RUN 1: 
![alt text](image.png)

This is sort of a logic dump so on my very first implementation i seemed to get crazy overfitting to the degree that i was getting super high val/training accuracy of like 99% but when it came to the test data i would just underperform majorly. I.e., that my implementation seemed to memorise training data but had failed on real data.

Plus based on my results from analysis it seemed to be an issue where I was overpredicting the number of normals (referred to as a type 2 errors so lots of false negatives); I was over predicting the amount of normal MRI results despite the fact they were all AD. This was the overwhelming source of error in my implementation.

My idea here is to increase the weighting for more significant focus on the AD (heavier penalty if i mistake the AD) and to have an increased no. of epochs as well as to increase my dropout (I am trying to fix my val accuracy vs test accuracy discrepancy).
===================================================================================================================================================================================
lOWKEY ALL THE training data is gaslightingb and lying to me it just does so shit on the final
RUN 2:
TEST SET RESULTS:
  Accuracy:  0.7423 ✗ Below target
  Precision: 0.9144
  Recall:    0.5296
  F1 Score:  0.6707
  AUC:       0.8281
  Total training time: 44.4 minutes
  Phase 1: 9.4 min | Phase 2: 35.0 min

===================================================================================================================================================================================
Run 3:
TEST SET RESULTS:
  Accuracy:  0.7516 ❌ Below target
  Precision: 0.8757
  Recall:    0.5812
  F1 Score:  0.6987
  AUC:       0.8447
  Total time: 36.4 minutes
  Phase 1: 13.5 min | Phase 2: 22.9 min
  Total epochs: 27
  Time per epoch: 80.8 seconds

  AGAIN the same issue here is the recall and overfitting where it will memorise training data and do really well there but when it comes to learning new datra it completely collapses herew sop the idea was now to change the weightings to more heavily penalise for an incorrect guessing as this will tailor the model to pay more attention to the AD cases, and I also have added an f2-score that is willing to sacrifice accuracy for the sake of a model that is more precise with finding the Ad cases in particular as that is our biggest cause for concern we seem to be missing up to 40% of the AD cases - classifying the sick as the healthy it is very concerneing. The numnber of epochs was increased to give the model a little more time to train as well as the learning rate decreases for a smoother more progressive appraoch to learning so it is not as unstable.

===================================================================================================================================================================================
  Run 4:
  OPTIMIZING DECISION THRESHOLD (Targeting Recall >= 0.85

TEST SET RESULTS:
  Accuracy:  0.7466 (✗)
  Recall:    0.5565 (✗)  <-- KEY METRIC
  Precision: 0.8912
  F1 Score:  0.6852
  F2 Score:  0.6017  <-- OPTIMIZED FOR
  AUC:       0.8207

⏱️Training Summary:
   Total time: 50.1 minutes
   Phase 1: 13.0 min | Phase 2: 37.0 min
   Total epochs: 37
   Time per epoch: 81.2 seconds

   so the new method recommended to me actually made it a lot worse lmao so it overfit pretty badly, the learning rate made itg more unstable and trhe focal loss alpha was too intense and the optimised f2-score and decreased the accuracy even further so let us bgin one more to get this right.

===================================================================================================================================================================================
 Run 5:
OPTIMIZING THRESHOLD (Target Recall ~0.82)
✓ Optimal threshold: 0.410
  Val metrics at threshold: Acc=0.9786, Recall=0.9929

🎯 TEST RESULTS:
  Accuracy:  0.7806 ❌
  Recall:    0.6244 ❌  <-- KEY METRIC
  Precision: 0.9028
  F1 Score:  0.7382
  F2 Score:  0.6655
  AUC:       0.8710

⏱️  Summary:
   Total time: 46.8 minutes
   Total epochs: 30
   Time per epoch: 93.5 seconds

   Hopeful new improvemnets on my newest run after run 5: 
   # 🏗️ ARCHITECTURE COMPARISON

My original 2 branch implementation:

```
┌─────────────────────────────────────┐
│      Input Image (224×224×3)        │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   ConvNeXt Backbone (Pretrained)    │
│   - Stem                            │
│   - Stage 0 (96 features)           │
│   - Stage 1 (192 features)          │
│   - Stage 2 (384 features)          │
│   - Stage 3 (768 features)          │
│   Output: [B, 1024, 7, 7]           │
└──────────────┬──────────────────────┘
               ↓
        ┌──────┴──────┐
        ↓             ↓
   ┌────────┐   ┌────────┐
   │  GAP   │   │  GMP   │
   │ Branch │   │ Branch │
   │   1    │   │   2    │
   └────┬───┘   └───┬────┘
        │           │
        │ [B,1024]  │ [B,1024]
        └─────┬─────┘
              ↓
       ┌──────────────┐
       │ Concatenate  │
       │  [B, 2048]   │  ← Only 2 branches!
       └──────┬───────┘
              ↓
       ┌──────────────┐
       │  Dropout     │
       │  Linear(512) │
       │  ReLU        │
       │  Dropout     │
       │  Linear(2)   │
       └──────┬───────┘
              ↓
         [B, 2] Output
```

improved 3 branch architecture suggestion:
```
┌─────────────────────────────────────┐
│      Input Image (224×224×3)        │
│   + CLAHE Enhancement (NEW!)        │  🔬
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│   ConvNeXt Backbone (Pretrained)    │
│   Same structure as before          │
│   Output: [B, 1024, 7, 7]           │
└──────────────┬──────────────────────┘
               ↓
      ┌────────┼────────┐
      ↓        ↓        ↓
┌─────────┐ ┌─────────┐ ┌──────────────┐
│   GAP   │ │   GMP   │ │  ATTENTION   │  ⭐ NEW!
│ Branch  │ │ Branch  │ │   Branch     │
│    1    │ │    2    │ │      3       │
├─────────┤ ├─────────┤ ├──────────────┤
│ Avg Pool│ │ Max Pool│ │ Learn where  │
│         │ │         │ │ to look!     │
│         │ │         │ │              │
│         │ │         │ │ Conv(1024->128)
│         │ │         │ │ ReLU         │
│         │ │         │ │ Conv(128->1) │
│         │ │         │ │ Sigmoid      │
│         │ │         │ │ ×Features    │
│         │ │         │ │ Avg Pool     │
└────┬────┘ └────┬────┘ └──────┬───────┘
     │           │             │
     │[B,1024]   │[B,1024]     │[B,1024]
     └─────┬─────┴──────────┬──┘
           ↓                ↓
     ┌──────────────────────────┐
     │      Concatenate         │
     │      [B, 3072]           │  ← 3 branches! 50% more!
     └────────────┬─────────────┘
                  ↓
     ┌──────────────────────────┐
     │  Feature Selection       │  ⭐ NEW!
     │  Linear(3072) + Sigmoid  │
     │  × Input (Gating)        │
     └────────────┬─────────────┘
                  ↓
     ┌──────────────────────────┐
     │  Enhanced Classifier     │
     │  ------------------------│
     │  Dropout(0.3)            │
     │  Linear(512) + BN + ReLU │  ✅ BatchNorm
     │  Dropout(0.15)           │
     │  Linear(256) + BN + ReLU │  ✅ Extra layer
     │  Dropout(0.1)            │
     │  Linear(2)               │
     └────────────┬─────────────┘
                  ↓
             [B, 2] Output
```

===================================================================================================================================================================================
Run 6: Hopefully better with this triple architecture"

.\train.py:525: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will exec
ute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This li
mits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`.
We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(Config.CHECKPOINT_DIR / 'best_model_final.pth')
Loaded best final model from epoch 17 (Val F2: 0.8701)
Optimizing threshold based on best Validation F2-Score...
✓ Optimal threshold: 0.480
  Val metrics at threshold: Acc=0.7153, Recall=0.9583

Evaluating on Test Set with optimal threshold...

🎯 TEST RESULTS:
  Accuracy:  0.6676 ❌
  Recall:    0.8123 ✅  <-- KEY METRIC
  Precision: 0.6270
  F1 Score:  0.7078
  F2 Score:  0.7670  <-- OPTIMIZED FOR
  AUC:       0.7617

⏱️  Summary:
   Total time: 25.7 minutes
   Phase 1: 14.2 min (12 epochs) | Phase 2: 11.5 min (8 epochs)
   Total epochs run: 20
   Time per epoch: 77.2 seconds

===================================================================================================================================================================================
Run 7: trying a new claude implementation with a bunch of suggestions based on what it did not like in my code:
Here are improvements based on recent literature that supposedly will assist in a high accuracy:
99.43% Accuracy - "A novel CNN architecture for accurate early detection" (El-Assy et al., 2024)

Used dual CNN branches with concatenation
Strong data augmentation


98.57% Accuracy - "Deep Learning-Based Ensemble Method" (2024)

Ensemble of 6 CNN models
Slice-based approach with entropy selection


96.5% Accuracy - "3D CNNs from intelligently selected neuroimaging" (2025)

Used 3D convolutions
Smart frame selection


95.93% Accuracy - "Lightweight Deep Learning Model on MRI Data" (2023)

Used attention mechanisms
Proper preprocessing pipeline

Why These Changes Work:
Recent studies emphasize that combining attention mechanisms, proper preprocessing (your CLAHE is perfect!), and multi-branch architectures significantly improves AD detection FrontiersarXiv. The CLAHE preprocessing you're already using combined with proper augmentation and attention mechanisms has been shown to be particularly effective Nature.
The key insight from the research is that models using multiple pooling strategies and attention mechanisms consistently achieve >90% accuracy Nature, which is why I've enhanced your 3-branch design to 4 branches with CBAM attention.


🎯 TEST RESULTS:
  Accuracy:  0.6612 ❌
  Recall:    0.4141 ❌
  Precision: 0.8090
  F1 Score:  0.5478
  F2 Score:  0.4589
  AUC:       0.7763

⏱️  Summary:
   Total time: 13.8 minutes
   Phase 1: 6.6 min | Phase 2: 7.3 min
   Total epochs: 25

================================================================================================================================================================

Run 8: 
  Accuracy:  0.6358 ⚠️
  Precision: 0.7456
  Recall:    0.4022
  F1 Score:  0.5226
  AUC:       0.7056
  📊 Confusion matrix saved to alzheimer_results_98\plots\confusion_matrix.png

  Run 9:
🎯 Performance:
   Accuracy:  0.7669 ❌
   Recall:    0.6962 ❌
   Precision: 0.8069
   F1 Score:  0.7475
   F2 Score:  0.7158
   AUC:       0.8428
   ⏱️  Total training time: 52.5 minutes
================================================================================================================================================================