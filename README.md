🧠 Food Waste Recognition & Estimation Using Deep Learning and Image Processing

Author: Shayan Rokhva
Supervisor: Dr. Babak Teimourpour
*Tarbiat Modares University (2025)

📘 Overview

This repository presents the full implementation of the master’s thesis “Food Waste Recognition and Estimation Using Deep Learning and Image Processing.” The project develops an intelligent framework capable of recognizing and quantifying food waste in realistic environments such as university dining halls and restaurants. Using deep learning and semantic segmentation, the system analyzes plate images taken before and after consumption and automatically estimates the amount of leftover food. The motivation behind this work arises from the growing global challenge of food waste, a phenomenon with enormous environmental, economic, and ethical implications. Traditional approaches—such as manual weighing of plates—are time-consuming, expensive, and impractical for large-scale operations. In contrast, the proposed framework provides a fast, accurate, and fully automated solution that leverages modern advances in computer vision and deep learning.

At its core, this project introduces a set of convolutional architectures (U-Net, U-Net++, and a customized lightweight variant) supported by a novel loss function and an improved evaluation pipeline. The system achieves Dice coefficients above 0.85 and a distributional pixel accuracy (DPA) greater than 0.9 across all classes, while sustaining inference speeds suitable for real-time deployment. This combination of performance and efficiency demonstrates the feasibility of intelligent vision-based monitoring for sustainable food management.

🎯 Objectives and Research Motivation

The central objective of this work is to design a practical deep learning pipeline that can estimate food waste automatically, accurately, and efficiently. Specifically, the research seeks to determine how semantic segmentation can be applied to distinguish food components on plates and measure the remaining portions. Beyond methodological goals, the study aims to contribute to the broader agenda of sustainable consumption by providing quantitative insights into food waste patterns. In doing so, it builds a bridge between artificial intelligence research and environmental applications, showing how vision-based models can inform data-driven decision-making in dining halls, catering systems, and food policy design.

🍽️ Dataset and Data Preparation

The dataset was created through direct field collection at a university dining hall. Each meal was photographed in two states: before consumption (the full plate) and after consumption (the same type of meal partially eaten). Because meals were standardized across all diners—identical portions and compositions—the average pre-consumption image served as a valid reference for estimating post-consumption waste. This design choice makes the dataset representative of real-world conditions, where tracking every single plate is not feasible.

Images were stored in separate folders for each food category, such as Adas Polo, Fesenjan, Chelo Goosht, Gheyme Bademjan, and Protein & Fries. Each folder contained the original .jpg image and its corresponding semantic mask in .png format. Pixel labels represented background and food components (for example, 0 = background, 1 = rice, 2 = stew). All images were resized to 256 × 256 pixels, striking a balance between visual detail and computational speed.

The labeling process was performed manually using the Roboflow platform, ensuring pixel-level accuracy. Masks identified broad semantic categories rather than individual food items, since the task focused on component-wise segmentation rather than instance counting. The dataset was divided into 70 % training, 15 % validation, and 15 % testing subsets, with the split performed before any augmentation to prevent data leakage. Comprehensive integrity checks guaranteed correct pairing of images and masks, consistent dimensions, and valid label ranges.

To improve generalization, data augmentation was applied exclusively to the training set. Techniques included random rotation, horizontal flipping, brightness and contrast adjustments, and small random crops. These transformations simulated realistic variations in lighting and camera angle while preserving the semantic content of the plates.

⚙️ Implementation Pipeline

All experiments were implemented in PyTorch. Random seeds were fixed for reproducibility, and the complete workflow was encapsulated in a single Jupyter notebook accompanied by modular Python scripts. The SegmentationDataset class dynamically loaded image–mask pairs, converted them to tensors, and normalized values. Data loaders for the training, validation, and testing sets were created with adjustable batch sizes to ensure efficient GPU utilization.

The codebase also included visualization utilities that allowed random sampling of images, overlays of predictions on ground truths, and generation of training-curve plots for loss, Dice, and IoU metrics. This visualization component was essential for verifying dataset correctness and for qualitatively assessing segmentation boundaries throughout training.

🏗️ Model Architectures

Four model configurations were developed and compared. The baseline was the U-Net, a fully convolutional encoder–decoder network characterized by skip connections that transmit spatial detail from encoder to decoder layers. This structure allows precise pixel-level localization, making it highly effective for segmentation tasks.

The second configuration, U-Net++, introduced nested skip connections and dense convolutional blocks that enable multi-scale feature fusion. It captures fine details and boundary information better than the standard U-Net, albeit with more parameters.

The third configuration was a lightweight U-Net, a custom design that halves the number of feature channels in each convolutional block. This drastically reduces parameter count and increases inference speed, making it ideal for real-time or embedded applications.

Finally, an optimized hybrid model was created that integrates the strengths of the previous variants. It maintains a lightweight encoder while introducing minor attention-like refinements and fine-tuned hyperparameters. This hybrid achieves nearly the same accuracy as U-Net++ but with significantly higher throughput.

🧮 Customized Loss Function

A central innovation of this project is the Capped Dynamic Weighted Cross-Entropy Loss (CDW-CEL), developed to counter severe class imbalance. In segmentation tasks of this nature, background pixels dominate the image, while smaller food regions—especially sauces or garnishes—represent minority classes. Standard cross-entropy would favor majority pixels, leading to biased learning. The proposed CDW-CEL assigns inverse-frequency weights to each class in every batch but caps their ratio at a maximum of ten. This “capping” mechanism prevents the instability that often occurs when extremely rare classes receive excessively large gradients. As a result, training remains stable, minority classes are learned effectively, and convergence is accelerated. The mathematical expression can be summarized as

𝑤
𝑖
=
min
⁡
(
1
𝑓
𝑖
,
1
𝑓
𝑚
𝑖
𝑛
×
cap
)
w
i
	​

=min(
f
i
	​

1
	​

,
f
min
	​

1
	​

×cap),
where 
𝑓
𝑖
f
i
	​

 denotes the relative frequency of class i.

🚀 Optimization Strategy

Training employed the AdamW optimizer, chosen for its decoupled weight-decay formulation and ability to converge quickly on noisy gradients. A dynamic learning-rate scheduler progressively reduced the rate every few epochs—typically every five—to encourage stable convergence. Regularization through weight decay further minimized overfitting. Typical batch sizes ranged from 8 to 16 depending on GPU capacity. Each training run spanned approximately 40–60 epochs, with checkpoints automatically stored whenever the validation IoU improved.

📏 Evaluation Metrics

Model performance was assessed using several complementary metrics. The Dice coefficient (or F1 score) measured overlap between prediction and ground truth, while Intersection-over-Union (IoU) provided a stricter pixel-wise agreement ratio. Pixel Accuracy calculated the overall proportion of correctly classified pixels but could be skewed by dominant classes. To address this, a new criterion called Distributional Pixel Accuracy (DPA) was proposed. DPA normalizes accuracy contributions across all classes by weighting each according to its relative frequency, producing a fairer reflection of segmentation quality in imbalanced datasets. Collectively, these metrics provide a comprehensive evaluation of both global accuracy and class-specific precision.

🔁 Training Procedure

The training loop followed the standard PyTorch paradigm. For each batch, images and masks were loaded, a forward pass produced predictions, the custom loss was computed, and gradients were back-propagated through the network. Parameters were then updated using AdamW, and the scheduler adjusted the learning rate as planned. Metrics were logged at each epoch, and model checkpoints were written based on the best validation IoU rather than loss alone.

After training, the optimal model was evaluated on the unseen test set, producing quantitative metrics and qualitative visualizations of segmentation masks. Training history—loss curves, Dice, IoU, and DPA trends—was saved for later comparison. This process ensured transparent and reproducible performance tracking.

💾 Results and Discussion

Experimental results confirmed that all models achieved reliable segmentation and waste estimation accuracy. The baseline U-Net reached a mean Dice of 0.85 and DPA ≈ 0.90 at around 40 frames per second. U-Net++ improved the Dice to 0.87 and IoU to 0.82 but ran slower at 32 FPS. The lightweight U-Net maintained competitive accuracy (Dice 0.84, IoU 0.78) while achieving real-time inference of ≈ 83 FPS. The optimized hybrid model achieved the best overall balance, with Dice 0.88, IoU 0.83, DPA 0.92, and ≈ 60 FPS inference speed.

These findings illustrate that carefully tuned architectures can deliver both high accuracy and computational efficiency. The combination of dynamic weighting, learning-rate scheduling, and architectural simplification allowed the system to converge stably while avoiding overfitting. Qualitative visualizations further confirmed precise segmentation boundaries and effective differentiation between food components, even under challenging lighting or occlusion conditions.

🧠 Analytical Insights

Several technical insights emerged from the study. First, integrating data-driven deep learning with image-based monitoring can fully automate the traditionally manual process of food-waste estimation. Second, semantic segmentation—particularly using encoder–decoder structures—captures spatial and compositional cues critical for accurate waste quantification. Third, the use of class-weighted and capped loss functions proved essential for learning balanced representations across categories of vastly different pixel distributions. Finally, architectural customization allowed a favorable trade-off between model complexity and real-time performance, opening the door to deployment on affordable hardware platforms.

🧰 Project Organization and Usage

The repository follows a clean modular structure. All raw and processed data are stored in the data/ directory, organized by food category and train–test partition. Model definitions and helper classes reside in scripts/, including model_unet.py, model_unetpp.py, dataset.py, loss_functions.py, and train.py. Trained models are saved in the models/ folder, and results such as qualitative samples, metric summaries, and training curves are collected under results/. The Jupyter notebook FoodWaste_Estimation.ipynb offers an end-to-end demonstration of the full workflow.

To reproduce experiments, users should install dependencies (torch, torchvision, albumentations, numpy, matplotlib, tqdm), clone the repository, and execute the training or evaluation scripts. Training commands allow the selection of architecture, number of epochs, and batch size. Evaluation scripts load the best model weights and compute test-set metrics. The notebook provides additional visualization for exploratory analysis.

🌍 Practical Applications

Beyond academic experimentation, this framework holds tangible real-world value. In large dining facilities, it can automatically monitor the amount of food wasted per meal type, guiding managers toward better menu design and portion control. In public institutions and restaurants, the system can act as a decision-support tool for resource planning and sustainability reporting. Integration with Internet-of-Things cameras or mobile devices would enable continuous real-time tracking. On a larger scale, the dataset and model can support policy studies by offering quantitative evidence of consumption behavior and waste reduction outcomes. The modular design also makes the framework adaptable to related domains such as agricultural product monitoring, food-quality inspection, or even nutrient estimation.

📚 Scientific Contributions

The research contributes to the literature in multiple dimensions. It presents one of the first comprehensive frameworks that combines semantic segmentation, customized loss design, and weighted evaluation for food-waste estimation using real-world data. It introduces the capped dynamic weighted loss, ensuring stable optimization in unbalanced scenarios, and proposes DPA as a fairer performance metric. The project also demonstrates how architectural simplification can lead to real-time inference without notable accuracy degradation. By providing both theoretical development and practical implementation, it bridges the gap between deep learning research and sustainable-food-system applications.

🔮 Future Work

Future extensions could explore attention mechanisms such as CBAM or SE blocks to enhance focus on relevant image regions. Expanding the dataset to include multi-angle or depth information would allow three-dimensional waste estimation rather than two-dimensional surface analysis. Another promising direction is integrating temporal data to analyze patterns of food consumption over time or deploying the lightweight version on embedded platforms like Raspberry Pi or NVIDIA Jetson. Cross-institutional collaborations could also help build a larger, standardized dataset for benchmarking semantic segmentation in the food domain.

🙏 Acknowledgments and Contact

This project was completed under the supervision of Dr. Babak Teimourpour, whose continuous guidance and support were invaluable. The author also thanks the students and staff who assisted in data collection and labeling.

For inquiries or collaboration opportunities:
Email: shayanrokhva1999@gmail.com

LinkedIn: linkedin.com/in/shayanrokhva

WhatsApp: +98 939 397 2774
