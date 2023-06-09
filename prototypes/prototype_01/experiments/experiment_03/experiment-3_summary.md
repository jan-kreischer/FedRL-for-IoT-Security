  
# Prototype 1 (Experiment 3)  
---  
  
Executed on 06.03.2023  
## Configuration  
### Server  
- nr\_clients: 2  
- nr\_rounds: 10  
- nr\_epochs\_per\_round: 1000  
- parallelized: True  
  
### Client 1  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.0001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 9612 samples of Behavior.NORMAL  
- 6205 samples of Behavior.RANSOMWARE\_POC  
- 3410 samples of Behavior.ROOTKIT\_BDVL  
- 4744 samples of Behavior.ROOTKIT\_BEURK  
### Client 2  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.0001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 9612 samples of Behavior.NORMAL  
- 4943 samples of Behavior.CNC\_THETICK  
- 2307 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3649 samples of Behavior.CNC\_OPT1  
- 2537 samples of Behavior.CNC\_OPT2  
### Global Agent  
- id: 0  
- batch\_size: 100  
- epsilon: 0  
- batch\_size: 46  
- batch\_size: 4  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 1/10  
  
- Training Round 1 on Client 1 took 14.8s  
- Training Round 1 on Client 2 took 8.63s  
![graph](round-01\_agent-01\_learning-curve.png)  
![graph](round-01\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.23%     | ransomware\_file\_extension\_hide |
| bdvl               | 100.00%    | rootkit\_sanitizer              |
| beurk              | 100.00%    | rootkit\_sanitizer              |
| the\_tick           | 0.00%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 0.00%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 0.00%      | rootkit\_sanitizer              |
| beurk              | 0.00%      | rootkit\_sanitizer              |
| the\_tick           | 100.00%    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 8.86%      | rootkit\_sanitizer              |
| beurk              | 99.11%     | rootkit\_sanitizer              |
| the\_tick           | 0.20%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.35%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.27%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 2/10  
  
- Training Round 2 on Client 1 took 6.61s  
- Training Round 2 on Client 2 took 6.7s  
![graph](round-02\_agent-01\_learning-curve.png)  
![graph](round-02\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 100.00%    | rootkit\_sanitizer              |
| beurk              | 100.00%    | rootkit\_sanitizer              |
| the\_tick           | 0.00%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 0.00%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 0.00%      | rootkit\_sanitizer              |
| beurk              | 0.00%      | rootkit\_sanitizer              |
| the\_tick           | 100.00%    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 0.00%      | rootkit\_sanitizer              |
| beurk              | 0.34%      | rootkit\_sanitizer              |
| the\_tick           | 99.61%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 97.96%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 3/10  
  
- Training Round 3 on Client 1 took 6.53s  
- Training Round 3 on Client 2 took 5.05s  
![graph](round-03\_agent-01\_learning-curve.png)  
![graph](round-03\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 100.00%    | rootkit\_sanitizer              |
| beurk              | 100.00%    | rootkit\_sanitizer              |
| the\_tick           | 0.00%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 0.00%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 0.00%      | rootkit\_sanitizer              |
| beurk              | 0.00%      | rootkit\_sanitizer              |
| the\_tick           | 100.00%    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 12.22%     | rootkit\_sanitizer              |
| beurk              | 99.59%     | rootkit\_sanitizer              |
| the\_tick           | 0.13%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.27%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 4/10  
  
- Training Round 4 on Client 1 took 4.95s  
- Training Round 4 on Client 2 took 4.18s  
![graph](round-04\_agent-01\_learning-curve.png)  
![graph](round-04\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 100.00%    | rootkit\_sanitizer              |
| beurk              | 100.00%    | rootkit\_sanitizer              |
| the\_tick           | 0.00%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 0.00%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 0.00%      | rootkit\_sanitizer              |
| beurk              | 0.00%      | rootkit\_sanitizer              |
| the\_tick           | 100.00%    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 34.37%     | rootkit\_sanitizer              |
| beurk              | 99.86%     | rootkit\_sanitizer              |
| the\_tick           | 0.00%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 91.85%     | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 5/10  
  
- Training Round 5 on Client 1 took 5.24s  
- Training Round 5 on Client 2 took 4.4s  
![graph](round-05\_agent-01\_learning-curve.png)  
![graph](round-05\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 100.00%    | rootkit\_sanitizer              |
| beurk              | 100.00%    | rootkit\_sanitizer              |
| the\_tick           | 0.00%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 0.00%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 0.00%      | rootkit\_sanitizer              |
| beurk              | 0.00%      | rootkit\_sanitizer              |
| the\_tick           | 100.00%    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 1.24%      | rootkit\_sanitizer              |
| beurk              | 92.74%     | rootkit\_sanitizer              |
| the\_tick           | 3.59%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 6.46%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.89%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 6/10  
  
- Training Round 6 on Client 1 took 5.08s  
- Training Round 6 on Client 2 took 5.4s  
![graph](round-06\_agent-01\_learning-curve.png)  
![graph](round-06\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 100.00%    | rootkit\_sanitizer              |
| beurk              | 100.00%    | rootkit\_sanitizer              |
| the\_tick           | 0.00%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 0.00%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 0.00%      | rootkit\_sanitizer              |
| beurk              | 0.00%      | rootkit\_sanitizer              |
| the\_tick           | 100.00%    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 1.42%      | rootkit\_sanitizer              |
| beurk              | 97.81%     | rootkit\_sanitizer              |
| the\_tick           | 0.52%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 2.35%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.44%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 7/10  
  
- Training Round 7 on Client 1 took 7.35s  
- Training Round 7 on Client 2 took 3.43s  
![graph](round-07\_agent-01\_learning-curve.png)  
![graph](round-07\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 100.00%    | rootkit\_sanitizer              |
| beurk              | 100.00%    | rootkit\_sanitizer              |
| the\_tick           | 0.00%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 0.00%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 0.00%      | rootkit\_sanitizer              |
| beurk              | 0.00%      | rootkit\_sanitizer              |
| the\_tick           | 100.00%    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 4.16%      | rootkit\_sanitizer              |
| beurk              | 99.04%     | rootkit\_sanitizer              |
| the\_tick           | 0.07%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.35%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.18%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 8/10  
  
- Training Round 8 on Client 1 took 4.88s  
- Training Round 8 on Client 2 took 4.17s  
![graph](round-08\_agent-01\_learning-curve.png)  
![graph](round-08\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 100.00%    | rootkit\_sanitizer              |
| beurk              | 100.00%    | rootkit\_sanitizer              |
| the\_tick           | 0.00%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 0.00%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 0.00%      | rootkit\_sanitizer              |
| beurk              | 0.00%      | rootkit\_sanitizer              |
| the\_tick           | 100.00%    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 23.03%     | rootkit\_sanitizer              |
| beurk              | 99.59%     | rootkit\_sanitizer              |
| the\_tick           | 0.00%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.09%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 99.88%     | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 9/10  
  
- Training Round 9 on Client 1 took 6.29s  
- Training Round 9 on Client 2 took 3.89s  
![graph](round-09\_agent-01\_learning-curve.png)  
![graph](round-09\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 100.00%    | rootkit\_sanitizer              |
| beurk              | 100.00%    | rootkit\_sanitizer              |
| the\_tick           | 0.00%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 0.00%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 0.00%      | rootkit\_sanitizer              |
| beurk              | 0.00%      | rootkit\_sanitizer              |
| the\_tick           | 100.00%    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 3.28%      | rootkit\_sanitizer              |
| beurk              | 98.63%     | rootkit\_sanitizer              |
| the\_tick           | 0.20%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 1.17%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.35%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 10/10  
  
- Training Round 10 on Client 1 took 6.63s  
- Training Round 10 on Client 2 took 4.12s  
![graph](round-10\_agent-01\_learning-curve.png)  
![graph](round-10\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 100.00%    | rootkit\_sanitizer              |
| beurk              | 100.00%    | rootkit\_sanitizer              |
| the\_tick           | 0.00%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 0.00%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 0.00%      | rootkit\_sanitizer              |
| beurk              | 0.00%      | rootkit\_sanitizer              |
| the\_tick           | 100.00%    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 16.12%     | rootkit\_sanitizer              |
| beurk              | 99.45%     | rootkit\_sanitizer              |
| the\_tick           | 0.00%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.12%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.18%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  

 ### Total training time: 145.53s  
