# Hair And Scalp Inspection System

## Project Description

The health of an individual can be impacted by common hair and scalp problems such as psoriasis, alopecia, head lice, and fungal infections. Conventional diagnostic techniques are frequently subjective, labor-intensive, and manual. To increase precision, effectiveness, and scalability in clinical settings, this project suggests an automated approach for diagnosing hair and scalp diseases using deep learning techniques.

The objective of this study is to create an automated system that can classify high-resolution photos of hair and scalp conditions using optimized Convolutional Neural Network (CNN) models. The system seeks to improve diagnostic accuracy and lessen the need for manual techniques by utilizing a dataset of 13,200 photos across 10 classes along with healthy hair.

The process involved pre-processing and augmentation of high-resolution RGB images to increase dataset variability. Hyperparameter adjustment was conducted to maximize the effectiveness of learning. Classification performance was enhanced using a fine-tuned CNN architecture with residual learning to efficiently capture complex patterns and textures in the photos. The system uses pre-trained and optimized CNN models for this task.

The model attained a 93% classification accuracy, with further learning improving its ability to identify complex patterns in the images. This method's excellent performance makes the illness detection system dependable and effective. This study demonstrates the promise of deep learning in dermatology by offering a productive, automated method for identifying disorders of the hair and scalp. In addition to lowering human error and increasing diagnostic accuracy, the technology provides a scalable way to incorporate AI into clinical procedures. It lays the groundwork for future advancements in automated medical diagnosis.



---

## Table of Contents
1. [Project Description](#project-description)
2. [How to Install and Run the Project](#how-to-install-and-run-the-project)
3. [Credits](#credits)
4. [License](#license)

---

## How to Install and Run the Project

### Prerequisites
- **Python**: Version 3.8 or higher
- **Libraries**: TensorFlow, Keras,Google GenerativeAI,Pillow
- **Web Framework**: Flask

### Installation Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Anshad-Aziz/Hair_Scalp_Disease_Flask.git
   cd Hair_Scalp_Disease_Flask
   ```
2.**Install the dependencies**:
  ```bash
 pip install -r requirements.txt
```
3.**Run the application**:
 ```bash
  python app.py
```


## Credits

- **Team Members**:
  - **Anshad Aziz** 
  - **B Mahesh Kumar Shetty** 
  - **Shruthi** 
  - **Srijan K** 

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
