---
layout: page
title: Offensive AI Course
permalink: /course/

---

<body style="background-color: #142326"></body>

<a href="https://offensive-ai-lab.github.io/"><svg id="d3banner"></svg></a>

{% include datavis_d3banner.html %}

Welcome to the course site for `Offensive AI` <script>document.write(new Date().getFullYear())</script> which I teach at BGU in the Department of Software and Information Systems Engineering.  Below you will find the course syllabus and more content will be added over the semester.





### Syllabus 

**Course Name**: Offensive AI<br/>
 **Course Name (Hebrew)**: בינה מלאכותית זדונית<br/>
 **Course Number**: TBA<br/>
 **Course Structure**: 3 hours of lectures weekly<br/>
 **Course Credits**: 3<br/>
 **Lecturer**: Dr. Yisroel Mirsky<br/>

#### Course Description:

Artificial intelligence (AI) has provided us with the ability to automate tasks, extract information from vast amounts of data, and synthesize media that is nearly indistinguishable from the real thing. However, positive tools can also be used for negative purposes. In particular, cyber adversaries can also use AI, but to enhance their attacks and expand their campaigns.

In this course we will learn about attacks against AI systems (adversarial machine learning) such as model poisoning, model inversion, membership inference, trojaning, and adversarial examples. We will also learn about attacks which use AI, such as deepfakes for facial reenactment and voice cloning, advanced spyware, autonomous bots, evasive malware, and the use of machine learning to detect software vulnerabilities. Finally, throughout the course we will learn how we can defend against these attacks and learn the best practices for developing systems which are robust against them too. 

#### Purpose of the Course:

The goal of the course is to learn (1) how AI is being used by malicious actors to exploit our AI systems and enhance their cyberattacks, and (2) how we can defend against these threats and develop safer systems.

#### Prerequisites:

At least one course in machine learning (e.g., 372.1.4951, 372.1.4952, 372.2.5910) or relevant experience in the subject. The course is open to students outside of the department on the basis of availability and faculty member recommendation.

#### Course Requirements:

- Attendance is required.
- Students must learn the course from the lectures and any provided written materials.
- Students will submit one practical exercise in Python (10% of the grade), and one project which will be presented in the final lecture (15% of the grade).
- The final exam is 75% of the grade. Passing the exam is required for passing the course. 

#### Lectures: 

(There may be small modifications)

`Week 1` Introduction to machine learning and offensive AI.

*Attacks on AI*

`Week 2` Adversarial Machine Learning I (Causative Attacks): <br/>
Dataset poisoning and fault attacks (e.g., neural trojans, defense evasion, allergy attacks, clustering attacks).   <br/>`Week 3` Adversarial Machine Learning II (Exploratory Attacks): <br/>
Adversarial examples, sponge examples, model inversion, membership inference, and parameter inference. <br/>`Week 4` Prevention and Mitigation of Adversarial Machine Learning <br/>`Week 5` Lab: Adversarial Machine Learning with libraries and Torch in Python

*Attacks using AI: Deepfakes*

`Week 6` Deepfakes I: <br/>
Ethics of deepfakes and Generative AI used for facial reenactment  <br/>`Week 7` Deepfakes II: <br/>
Generative AI used for face replacement, face synthesis, and record tampering <br/>`Week 8` Deepfakes III: <br/>
Generative AI for voice cloning, spoofing, and audio driven reenactment  <br/>`Week 9` Detection, Prevention, and Mitigation of Deepfakes <br/>

*Attacks using AI: Attack Tools*

`Week 10` Attack Planning and Exploit Development  <br/>`Week 11` Spyware and Credential Theft <br/>`Week 12` Intelligent Bots, Swarms, Detection Evasion, and Campaign Automation

*Course Conclusion*

`Week 13` Student project presentations

#### Reading List: 

(tentative)

1. Huang, Ling, et al. "Adversarial machine learning." *Proceedings of the 4th ACM workshop on Security and artificial intelligence*. 2011.
2. Biggio, Battista, and Fabio Roli. "Wild patterns: Ten years after the rise of adversarial machine learning." *Pattern Recognition* 84 (2018): 317-331.
3. Zhang, Jiliang, and Chen Li. "Adversarial examples: Opportunities and challenges." *IEEE transactions on neural networks and learning systems* 31.7 (2019): 2578-2593.
4. Carlini, Nicholas, et al. "On evaluating adversarial robustness." *arXiv preprint arXiv:1902.06705* (2019).
5. Ilyas, Andrew, et al. "Adversarial examples are not bugs, they are features." *arXiv preprint arXiv:1905.02175* (2019).
6. Liu, Yuntao, et al. "A survey on neural trojans." *2020 21st International Symposium on Quality Electronic Design (ISQED)*. IEEE, 2020.
7. Chen, Huili, et al. "DeepInspect: A Black-box Trojan Detection and Mitigation Framework for Deep Neural Networks." *IJCAI*. 2019.
8. Mirsky, Yisroel, and Wenke Lee. "The creation and detection of deepfakes: A survey." *ACM Computing Surveys (CSUR)* 54.1 (2021): 1-41.
9. Tolosana, Ruben, et al. "Deepfakes and beyond: A survey of face manipulation and fake detection." *Information Fusion* 64 (2020): 131-148.
10. Arik, Sercan O., et al. "Neural voice cloning with a few samples." *arXiv preprint arXiv:1802.06006* (2018).
11. Hettwer, Benjamin, Stefan Gehrer, and Tim Güneysu. "Applications of machine learning techniques in side-channel attacks: a survey." *Journal of Cryptographic Engineering* 10.2 (2020): 135-162.
12. Batina, Lejla, et al. "{CSI}{NN}: Reverse Engineering of Neural Network Architectures Through Electromagnetic Side Channel." *28th {USENIX} Security Symposium ({USENIX} Security 19)*. 2019.