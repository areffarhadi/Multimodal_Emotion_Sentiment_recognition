# Multimodal_Emotion_Sentiment_recognition
Emotion Recognition using Multimodal vector on MELD dataset


```mermaid
flowchart LR
    A[Raw .npy Files <br>(Audio+Video)] --> B(SpeakerDataset)
    B --> C{DataLoader <br>+ <br>collate_fn}
    C --> D[MambaSpeakerRecognitionModel]
    D --> E[train_model(...)]
    
    subgraph MambaSpeakerRecognitionModel
        direction TB
        
        subgraph Input Projections
          A1(Audio Projection)
          A2(Video Projection)
        end
        
        subgraph Mamba Blocks
          M1(MambaBlock x N for Audio)
          M2(MambaBlock x N for Video)
        end
        
        subgraph Fusion
          F1(Mean Pool Audio)
          F2(Mean Pool Video)
          F3(Concatenate)
          F4(Joint MambaBlock)
        end
        
        subgraph Outputs
          O1(Embedding Layer)
          O2(Speaker Classifier)
          O3(Modality Classifier)
          O4(Contrastive Projection)
        end
        
        A1 --> M1
        A2 --> M2
        M1 --> F1
        M2 --> F2
        F1 --> F3
        F2 --> F3
        F3 --> F4
        F4 --> O1
        O1 --> O2
        O1 --> O3
        O1 --> O4
        
    end
    
    E -->|Loop Batches| G[Losses:<br>CrossEntropy + <br>Contrastive + <br>Modality]
    G --> H(Backward + Optimization)
    H --> I[Validation + Saving Best Model]
```
