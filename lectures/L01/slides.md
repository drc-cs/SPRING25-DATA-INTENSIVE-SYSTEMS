---
title: MBAI XXX
separator: <!--s-->
verticalSeparator: <!--v-->
theme: serif
revealOptions:
  transition: 'none'
---

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 50%; position: absolute;">

  # Data Intensive Systems
  ## L.01 | Introduction

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 10%">

  <iframe src="https://lottie.host/embed/216f7dd1-8085-4fd6-8511-8538a27cfb4a/PghzHsvgN5.lottie" height = "100%" width = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Welcome to Data Intensive Systems.
  ## Please check in by creating an account and entering the code on the chalkboard.

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 40%; padding-top: 5%">
    <iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=Check In" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

# Syllabus

<div style="overflow-y: scroll; height: 80%; font-size: 0.8em;">


</div>

<!--s-->

## Attendance

Attendance at lectures is **mandatory** and in your best interest. Your Attendance & Comprehension score is worth 20% of your final grade. Lectures will have graded quizzes throughout, and the top 12 scores will be used to calculate your attendance grade. This is designed to reduce the weight of exams and technical homeworks.

<!--s-->

## Grading

There is a high emphasis on the practical application of the concepts covered in this course. 
<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

| Component | Weight |
| --- | --- |
| Attendance & Comprehension | 40% |
| Homework | 40% |
| Exam Part I | 10% |
| Exam Part II | 10% |

</div>
<div class="c2" style = "width: 50%">

| Grade | Percentage |
| --- | --- |
| A | 94-100 |
| A- | 90-93 |
| B+ | 87-89 |
| B | 83-86 |
| B- | 80-82 |
| C+ | 77-79 |
| C | 73-76 |
| C- | 70-72 |

</div>
</div>

<!--s-->

## Homework 

Homeworks are designed to reinforce the concepts covered in lecture. They will be a mix of theoretical and practical problems, and each will include a programmatic and free response portion. If there is time, we will work on homework at the end of every lecture as a group. For the automated grading server, you get 3 attempts per homework assignment. The highest score will be recorded.

- **Due**: Homework due dates are posted in the syllabus.

- **Late Policy**: Late homeworks will lose 1 out of 10 points per day (1% of your final grade).

- **Platform**: A submission script has been provided to submit your homeworks.

- **Collaboration**: You are encouraged to work with your peers on homeworks. However, you must submit your own work. Copying and pasting code from other sources will be detected and penalized.

<!--s-->

## LLMs (The Talk)

<iframe src="https://lottie.host/embed/e7eb235d-f490-4ce1-877a-99114b96ff60/OFTqzm1m09.json" height = "100%" width = "100%"></iframe>

<!--s-->

## Exams

There are two exams in this class. They will cover the theoretical and practical concepts covered in the lectures and homeworks. If you follow along with the lectures and homeworks, you will be well-prepared for the exams.

<!--s-->

## Academic Integrity [&#x1F517;](https://www.northwestern.edu/provost/policies-procedures/academic-integrity/index.html)

### Homeworks

- Do not exchange code fragments on any assignments.
- Do not copy solutions from any source, including the web or previous CS 326 students.
- You cannot upload / sell your assignments to code sharing websites.

<!--s-->

## Accommodations

Any student requesting accommodations related to a disability or other condition is required to register with AccessibleNU and provide professors with an accommodation notification from AccessibleNU, preferably within the first two weeks of class. 

All information will remain confidential.

<!--s-->

## Mental Health

If you are feeling distressed or overwhelmed, please reach out for help. Students can access confidential resources through the Counseling and Psychological Services (CAPS), Religious and Spiritual Life (RSL) and the Center for Awareness, Response and Education (CARE).

<!--s-->

## Stuck on Something?

### **Office Hours**

- Time: TBA
- Location: TBA

### **Canvas Discussion**

- Every homework & project will have a discussion thread on Canvas.
- Please post your questions there so that everyone can benefit from the answers!

<!--s-->

## Stuck on Something?

### **Email**

We are here to help you! Please try contacting us through office hours or the dedicated Canvas discussion threads.

<div class = "col-wrapper" style="font-size: 0.8em;">
<div class="c1" style = "width: 50%">

**Peer Mentors**:

</div>
<div class="c2" style = "width: 50%">

**TA**:
- TBA

**Intructor**:
- Joshua D'Arcy: joshua.darcy@northwestern.edu

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Spring Quarter Plan
  ## After looking at the syllabus, is there anything you want me to cover that I'm not?

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 40%; padding-top: 5%">
    <iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=Coverage" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class="header-slide">

# Homework Assignment
## H.01 "Hello World!"

Due: 01.14.2025

</div>

<!--s-->

## H.01 | Docker Installation

The first step in this course is to install Docker. Docker is a platform for developing, shipping, and running applications in containers. We'll go into more detail about what Docker is and how it works in a future lecture.

In the meantime, please install Docker on your machine. You can find the installation instructions [here](https://docs.docker.com/get-docker/).

<!--s-->

## H.01 | Download Docker Image and Run Locally

Once you have Docker installed, you can download the Docker image for this course by running the following command in your terminal:

```bash

docker pull joshuadarcy/drc-cs-9a3f6:latest

```

Once the image is downloaded, you can run it using the following command:

```bash

docker run -p 8888:8888 joshuadarcy/drc-cs-9a3f6:latest

```

<!--s-->

## H.01 | Docker & Code-Server

The Docker image you just downloaded contains a code-server instance that you can access in your browser. This setup allows you to code in a Jupyter notebook environment without having to install anything (besides Docker) on your local machine. To access the code-server instance, open your browser and navigate to `localhost:8888`. You can bookmark this port for easy access in the future.

<!--s-->

## H.01 | Pulling

Before you start working on any homework, make sure you have the latest version of the repository. The following command in the integrated terminal will pull the latest version of the repository and give you access to the most up-to-date homework:

```bash
git pull
```

If you have any issues with using this git-based system, please reach out to us during office hours or via email.

<!--s-->

## H.01 | Opening Homework

Open the <span class="code-span">homeworks/</span> folder in VSCode. You should see a folder called <span class="code-span">H.01/</span>. Open the folder and you will see three files: 

- <span class="code-span">hello_world.py</span>: This file contains placeholders for the methods you will write.

- <span class="code-span">hello_world.ipynb</span>: This is a Jupyter notebook that provides a useful narrative for the homework and methods found in the <span class="code-span">hello_world.py</span> file.

- <span class="code-span">hello_world_test.py</span>: This is the file that will be used to test your code. Future homeworks will not include this file, and this is for demonstration purposes only.

Let's do the first homework together.

<!--s-->

<div class = "header-slide">

 ## Homework Demonstration

</div>

<!--s-->

## H.01 | Submitting Homework

You will submit your homework using the provided submission script. You should use the same username and password you used for the registration process earlier. You will usually receive your score within a few seconds, but this may take longer as the homeworks get more involved.

```bash
python submit.py --homework H01/hello_world.py --username your_username --password your_password 
```

<!--s-->

## H.01 | Homework Grading

The highest score will be recorded, so long as it is submitted before the deadline! You have 3 attempts for every homework. Late homeworks will be penalized 10% per day.

<!--s-->

<div class="header-slide">

# Questions?

</div>

<!--s-->