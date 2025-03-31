---
title: MBAI 417
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

Attendance at lectures is mandatory and in your best interest. 

Your **Attendance & Comprehension** score is worth 40% of your final grade. Lectures will have graded quizzes throughout, and the top 12 scores will be used to calculate your grade.

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

Homeworks are designed to reinforce the concepts covered in lecture. They will be a mix of theoretical and practical problems, and each will include a programmatic and free response portion. If there is time, we will work on homework at the end of every lecture as a group.

- **Due**: Homework due dates are posted in the syllabus.

- **Late Policy**: Late homeworks will lose 1 out of 10 points per day (1% of your final grade).

- **Platform**: A submission script has been provided to submit your homeworks to Canvas, which will be at the end of each notebook.

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
- Do not copy solutions from any source, including the web or previous MBAI students.
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

- Time: By Appointment
- Location: Mudd 3510

### **Canvas Discussion**

- Every homework & project will have a discussion thread on Canvas.
- Please post your questions there so that everyone can benefit from the answers! I will not respond to homework questions via email.

<!--s-->

## Stuck on Something?

### **Email**

We are here to help you! Please try contacting us through office hours or the dedicated Canvas discussion threads.

**TA**:
- TBA

**Intructor**:
- Joshua D'Arcy: joshua.darcy@northwestern.edu


<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Quarter Plan
  ## After looking at the syllabus, is there anything you want me to cover that I'm not?

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 40%; padding-top: 5%">
    <iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=Coverage" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class="header-slide">

# Homework Onboarding
## H.01 | "Hello, World!"

Due: 

</div>

<!--s-->

## H.01 | Docker Installation

The first step in this course is to install Docker. Docker is a platform for developing, shipping, and running applications in containers. We'll go into more detail about what Docker is and how it works in a future lecture.

In the meantime, please install Docker on your machine if you haven't already done so. You can find the installation instructions [here](https://docs.docker.com/get-docker/).

<!--s-->

## H.01 | Download Docker Image and Run Locally

Once you have Docker installed, you can download the Docker image for this course by running the following command in your terminal:

```bash
docker pull joshuadrc/mbai:latest
```

Once the image is downloaded, you can run it using the following command:

```bash
docker run -p 8888:8888 joshuadrc/mbai:latest
```

<!--s-->

## H.01 | Setting up Docker & Code-Server

The Docker image you just downloaded contains a [code-server](https://github.com/coder/code-server) instance that you can access in your browser. 

This setup allows you to code in a Jupyter notebook environment without having to install anything (besides Docker) on your local machine. To access the code-server instance, open your browser and navigate to <span class="code-span">http://localhost:8888</span>. 

You can bookmark this page for easy access in the future.

> **Note**: If you restart your computer, you will need to restart the Docker container via the application.

<!--s-->

## H.01 | Pulling

Before you start working on any homework, make sure you have the latest version of the repository. The following command in the integrated code-server terminal will pull the latest version of the repository and give you access to the most up-to-date homework:

```bash
git pull
```

<!--s-->

## H.01 | Updating your <span class="code-span">.env</span> file

The <span class="code-span">.env</span> file is a configuration file that contains environment variables for your application. It is used to store sensitive information such as API keys, database credentials, and other configuration settings.

Using the **same username and password that you used for the slides login**, please fill out the .env file located in the "homeworks" folder of your code-server instance. This information will be used to connect your homeworks to Canvas.

```bash
AG_USERNAME = "your_northwestern_email"
AG_PASSWORD = "your_password"
```

<!--s-->

## H.01 | Setting up Snowflake

Snowflake is a cloud-based data warehousing service that we will be using for this course. You will need to create an account and set up a database to use Snowflake. 

Please follow the instructions in the [Snowflake Setup Guide](https://signup.snowflake.com/) to create an account. You will also want to do the following once you have access: 

1. Set a budget limit. You get $400 in free credits for 30 days. Set budget cap to $20, just in case.
2. Collect your account identifer  <span class="code-span">Profile > Account > View Account Details > Account Identifier</span>
3. Collect your username and password. Snowflake requires MFA now, but you will still need to have a password.

<!--s-->

## H.01 | Updating your <span class="code-span">.env</span> file

Please fill out the following fields in the .env file using the information we just collected from Snowflake (this will allow you to connect to Snowflake from your laptop):

```bash
SNOWFLAKE_ACCOUNT = "your_account_identifier"
SNOWFLAKE_USER = "your_username"
SNOWFLAKE_PASSWORD = "your_password"
```

<!--s-->

<div class = "header-slide">

 # H.01 Demonstration

</div>

<!--s-->