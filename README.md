# A brief Summary of the Visual-Storyteller:

#### The model deployed in this project is a chained framework that creates stories for a series of images (task commonly known as Visual Storytelling). In order to do so, this task is dealt as a two-step problem, where first a captioner model creates descriptions for capturing the existential information in the sequence of images and a storyteller uses and reformulates these captions to a semantically enriched and coherent story.  

![My Image](Images/Model_diagram.png)

# How to Use:

__Step 1__: Install all the required packages from [the requirements file](./requirements.txt) using `pip install -r requirements.txt`.

__Step 2__: Download the [models_to_use](./models_to_use) folder and store it locally (~2GB). 

__Step 3__: Download the files [helper.py](./Evaluate/helper.py) and [evaluate.py](./Evaluate/evaluate.py) from the folder [Evaluate](./Evaluate) and store them locally under the same directory. 

Alternatively, use the following commands to download them directly:

```bash
# Download helper.py
wget https://raw.githubusercontent.com/username/repository/main/Evaluate/helper.py

# Download evaluate.py
wget https://raw.githubusercontent.com/username/repository/main/Evaluate/evaluate.py
```

__Step 4__: Create a folder locally with five images of your choice or pull one of the folders present under the folder with name 'Visual Stories'.

__Step 5__: Make the necessary adjustments (regarding the models/images paths) on the file 'evaluate.py' and run the file.



# Results: 

#### Some examples of generated stories are presented below: 

![My Image](Images/Story_example.png)

