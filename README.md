Multiple Instance Learning (MIL) has found widespread applications in the classification of whole slide images (WSIs) in pathology. Moreover, recent research indicates that MIL methods combined with feature fusion mechanisms often yield better results. However, this approach introduces redundant feature interference while integrating comprehensive pathological image features, leading to a decrease in performance. To address this issue, we have devised an Attention-based Feature Distillation Multi-Instance Learning (AFD-MIL) approach. This approach utilizes attention mechanisms to distill more valuable features and apply them to WSI classification. Additionally, we introduce global loss optimization to finely control the feature distillation module. We have implemented AFD-MIL on the Camelyon16 and NSCLC datasets, tailoring feature distillation methods for different diseases, resulting in enhanced performance and interpretability. AFD-MIL stands orthogonal to many existing MIL methods, bringing consistent performance improvements to them. It has surpassed the current state-of-the-art (SOTA) on both datasets, achieving an accuracy of 91.47% and an AUC of 94.29% in breast cancer classification and an accuracy of 93.33% and an AUC of 98.17% in non-small cell lung cancer classification.
![Uploading AFD-MIL_G.png…]()


Before starting to train the model, slice the Whole Slide Images (WSIs) and extract features by referring to the method of CLAM. Subsequently, assign the train.csv and test.csv files on our own to partition the data:
![image](https://github.com/user-attachments/assets/b83629c6-9ba9-41c5-a80b-0ae472258a58)
