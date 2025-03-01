### **Pest Detection W**  
**Real-time Pest Detection System using RAG, ResNet & Raspberry Pi**  

#### **Overview**  
Pest Detection W is an automated real-time pest detection system that helps farmers monitor their crops efficiently. The system captures images every 4 hours using a field camera connected to a Raspberry Pi, processes them using a ResNet model for pest identification, and utilizes a RAG (Retrieval-Augmented Generation) model to generate necessary pre-requisites and alerts for farmers.  

#### **Features**  
- **Automated Image Capture**: A Raspberry Pi-powered camera captures images at regular intervals.  
- **Pest Identification**: A ResNet model processes images to detect and classify pests.  
- **Intelligent Alert System**: The RAG model generates recommendations and alerts based on detected pests.  
- **Hardware Integration**: Works seamlessly with Raspberry Pi for real-time monitoring.  

#### **System Workflow**  
1. Raspberry Pi captures an image every 4 hours.  
2. The captured image is sent to the ResNet model for pest classification.  
3. If pests are detected, the RAG model generates recommendations.  
4. The system sends an alert to the farmer with necessary pre-requisites.  

#### **Installation & Setup**  
1. **Hardware Requirements**  
   - Raspberry Pi (Model 3B+ or higher)  
   - Camera Module (Raspberry Pi Camera v2 or higher)  
   - MicroSD card (16GB or higher)  
   - Power Supply  

2. **Software Requirements**  
   - Python 3.8+  
   - TensorFlow / PyTorch (for ResNet)  
   - OpenCV  
   - FastAPI / Flask (for API-based communication)  
   - RAG model dependencies (Hugging Face Transformers, FAISS, etc.)  

3. **Installation Steps**  
   ```sh
   git clone https://github.com/your-repo/pest-detection-w.git
   cd pest-detection-w
   pip install -r requirements.txt
   ```  

4. **Running the System**  
   ```sh
   python main.py
   ```  

#### **Demo Video**  
📽️ [Watch the Demo](https://drive.google.com/drive/folders/1Qib7GxLfwtaCiBk26SPo1OZgZnz7XetB?usp=drive_link)  

#### **Contributors**  
- Joel Ebenezer 
- Koushik Balaji 
- Shree Pranav

#### **License**  
MIT License  

