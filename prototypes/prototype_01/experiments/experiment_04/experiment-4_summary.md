  
# Prototype 1 (Experiment 4)  
---  
  
Executed on 06.03.2023  
## Configuration  
### Server  
- nr\_clients: 2  
- nr\_rounds: 10  
- nr\_epochs\_per\_round: 1000  
- parallelized: False  
  
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
- 9598 samples of Behavior.NORMAL  
- 6181 samples of Behavior.RANSOMWARE\_POC  
- 3406 samples of Behavior.ROOTKIT\_BDVL  
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
- 9598 samples of Behavior.NORMAL  
- 4952 samples of Behavior.CNC\_THETICK  
- 2380 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3655 samples of Behavior.CNC\_OPT1  
- 2544 samples of Behavior.CNC\_OPT2  
### Global Agent  
- id: 0  
- batch\_size: 100  
- epsilon: 0  
- batch\_size: 46  
- batch\_size: 4  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 1/10  
  
- Training Round 1 on Client 1 took 6.14s  
- Training Round 1 on Client 2 took 6.73s  
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
| bdvl               | 0.00%      | rootkit\_sanitizer              |
| beurk              | 33.95%     | rootkit\_sanitizer              |
| the\_tick           | 71.00%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 73.36%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 65.48%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 2/10  
  
- Training Round 2 on Client 1 took 5.85s  
- Training Round 2 on Client 2 took 6.2s  
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
| beurk              | 0.00%      | rootkit\_sanitizer              |
| the\_tick           | 100.00%    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 3/10  
  
- Training Round 3 on Client 1 took 9.89s  
- Training Round 3 on Client 2 took 6.92s  
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
| bdvl               | 7.44%      | rootkit\_sanitizer              |
| beurk              | 97.95%     | rootkit\_sanitizer              |
| the\_tick           | 0.39%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.59%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.71%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 4/10  
  
- Training Round 4 on Client 1 took 6.86s  
- Training Round 4 on Client 2 took 6.56s  
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
| bdvl               | 0.44%      | rootkit\_sanitizer              |
| beurk              | 76.18%     | rootkit\_sanitizer              |
| the\_tick           | 16.39%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 28.52%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 15.00%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 5/10  
  
- Training Round 5 on Client 1 took 7.36s  
- Training Round 5 on Client 2 took 4.57s  
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
| bdvl               | 0.80%      | rootkit\_sanitizer              |
| beurk              | 95.48%     | rootkit\_sanitizer              |
| the\_tick           | 1.05%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 3.99%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 1.24%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 6/10  
  
- Training Round 6 on Client 1 took 5.79s  
- Training Round 6 on Client 2 took 5.96s  
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
| bdvl               | 0.71%      | rootkit\_sanitizer              |
| beurk              | 95.55%     | rootkit\_sanitizer              |
| the\_tick           | 0.85%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 4.34%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 1.51%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 7/10  
  
- Training Round 7 on Client 1 took 5.6s  
- Training Round 7 on Client 2 took 3.76s  
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
| bdvl               | 10.01%     | rootkit\_sanitizer              |
| beurk              | 98.56%     | rootkit\_sanitizer              |
| the\_tick           | 0.07%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.23%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.44%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 8/10  
  
- Training Round 8 on Client 1 took 5.06s  
- Training Round 8 on Client 2 took 3.59s  
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
| bdvl               | 56.78%     | rootkit\_sanitizer              |
| beurk              | 99.86%     | rootkit\_sanitizer              |
| the\_tick           | 0.00%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 3.53%      | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 9/10  
  
- Training Round 9 on Client 1 took 6.7s  
- Training Round 9 on Client 2 took 5.77s  
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
| bdvl               | 12.22%     | rootkit\_sanitizer              |
| beurk              | 98.56%     | rootkit\_sanitizer              |
| the\_tick           | 0.07%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.23%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.44%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 10/10  
  
- Training Round 10 on Client 1 took 6.62s  
- Training Round 10 on Client 2 took 3.79s  
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
| bdvl               | 29.94%     | rootkit\_sanitizer              |
| beurk              | 99.38%     | rootkit\_sanitizer              |
| the\_tick           | 0.07%      | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 0.00%      | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 0.27%      | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 99.76%     | cnc\_ip\_shuffle                 |  

 ### Total training time: 152.0s  
