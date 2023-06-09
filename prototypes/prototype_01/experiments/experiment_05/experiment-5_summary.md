  
# Prototype 1 (Experiment 5)  
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
- 3655 samples of Behavior.CNC\_OPT1  
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
- 4744 samples of Behavior.ROOTKIT\_BEURK  
- 2544 samples of Behavior.CNC\_OPT2  
- 2380 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
### Global Agent  
- id: 0  
- batch\_size: 100  
- epsilon: 0  
- batch\_size: 46  
- batch\_size: 4  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 1/10  
  
- Training Round 1 on Client 1 took 9.72s  
- Training Round 1 on Client 2 took 10.07s  
![graph](round-01\_agent-01\_learning-curve.png)  
![graph](round-01\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.07%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.94%     | rootkit\_sanitizer              |
| beurk              | 4.52%      | rootkit\_sanitizer              |
| the\_tick           | 94.97%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 94.84%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 2.55%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 0.44%      | rootkit\_sanitizer              |
| beurk              | 20.67%     | rootkit\_sanitizer              |
| the\_tick           | 94.64%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 84.98%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 81.28%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 76.88%     | rootkit\_sanitizer              |
| beurk              | 0.68%      | rootkit\_sanitizer              |
| the\_tick           | 99.61%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 99.18%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 92.09%     | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 2/10  
  
- Training Round 2 on Client 1 took 9.12s  
- Training Round 2 on Client 2 took 9.17s  
![graph](round-02\_agent-01\_learning-curve.png)  
![graph](round-02\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 98.76%     | rootkit\_sanitizer              |
| beurk              | 5.48%      | rootkit\_sanitizer              |
| the\_tick           | 95.89%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 93.54%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 1.34%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 11.16%     | rootkit\_sanitizer              |
| beurk              | 80.42%     | rootkit\_sanitizer              |
| the\_tick           | 88.70%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 37.09%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 70.81%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 90.79%     | rootkit\_sanitizer              |
| beurk              | 20.12%     | rootkit\_sanitizer              |
| the\_tick           | 94.45%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 82.39%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.91%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 80.17%     | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 3/10  
  
- Training Round 3 on Client 1 took 8.32s  
- Training Round 3 on Client 2 took 9.02s  
![graph](round-03\_agent-01\_learning-curve.png)  
![graph](round-03\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 98.85%     | rootkit\_sanitizer              |
| beurk              | 3.90%      | rootkit\_sanitizer              |
| the\_tick           | 98.24%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 94.60%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 1.82%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 13.55%     | rootkit\_sanitizer              |
| beurk              | 32.03%     | rootkit\_sanitizer              |
| the\_tick           | 96.67%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 81.34%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 85.54%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 59.88%     | rootkit\_sanitizer              |
| beurk              | 16.36%     | rootkit\_sanitizer              |
| the\_tick           | 96.54%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 84.86%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.82%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 68.49%     | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 4/10  
  
- Training Round 4 on Client 1 took 7.65s  
- Training Round 4 on Client 2 took 5.93s  
![graph](round-04\_agent-01\_learning-curve.png)  
![graph](round-04\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 98.94%     | rootkit\_sanitizer              |
| beurk              | 1.64%      | rootkit\_sanitizer              |
| the\_tick           | 99.48%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 98.24%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.82%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 0.36%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 13.91%     | rootkit\_sanitizer              |
| beurk              | 48.05%     | rootkit\_sanitizer              |
| the\_tick           | 97.65%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 71.71%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 97.07%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 41.54%     | rootkit\_sanitizer              |
| beurk              | 9.03%      | rootkit\_sanitizer              |
| the\_tick           | 99.22%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 92.02%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 87.47%     | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 5/10  
  
- Training Round 5 on Client 1 took 5.16s  
- Training Round 5 on Client 2 took 5.73s  
![graph](round-05\_agent-01\_learning-curve.png)  
![graph](round-05\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 98.94%     | rootkit\_sanitizer              |
| beurk              | 3.76%      | rootkit\_sanitizer              |
| the\_tick           | 99.41%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 96.13%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.82%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 0.24%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 28.52%     | rootkit\_sanitizer              |
| beurk              | 74.13%     | rootkit\_sanitizer              |
| the\_tick           | 96.80%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 52.93%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 94.68%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 48.45%     | rootkit\_sanitizer              |
| beurk              | 22.11%     | rootkit\_sanitizer              |
| the\_tick           | 98.43%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 82.16%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 57.18%     | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 6/10  
  
- Training Round 6 on Client 1 took 5.07s  
- Training Round 6 on Client 2 took 4.35s  
![graph](round-06\_agent-01\_learning-curve.png)  
![graph](round-06\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 99.03%     | rootkit\_sanitizer              |
| beurk              | 4.18%      | rootkit\_sanitizer              |
| the\_tick           | 99.48%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 95.77%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 0.00%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 31.80%     | rootkit\_sanitizer              |
| beurk              | 65.50%     | rootkit\_sanitizer              |
| the\_tick           | 97.58%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 60.80%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 92.55%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 49.34%     | rootkit\_sanitizer              |
| beurk              | 20.88%     | rootkit\_sanitizer              |
| the\_tick           | 98.63%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 83.57%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 46.47%     | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 7/10  
  
- Training Round 7 on Client 1 took 5.38s  
- Training Round 7 on Client 2 took 4.56s  
![graph](round-07\_agent-01\_learning-curve.png)  
![graph](round-07\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 99.47%     | rootkit\_sanitizer              |
| beurk              | 5.00%      | rootkit\_sanitizer              |
| the\_tick           | 99.48%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 94.01%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 0.00%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 25.16%     | rootkit\_sanitizer              |
| beurk              | 59.00%     | rootkit\_sanitizer              |
| the\_tick           | 98.04%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 71.48%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 96.98%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 51.11%     | rootkit\_sanitizer              |
| beurk              | 23.20%     | rootkit\_sanitizer              |
| the\_tick           | 98.82%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 84.51%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 49.51%     | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 8/10  
  
- Training Round 8 on Client 1 took 6.9s  
- Training Round 8 on Client 2 took 4.64s  
![graph](round-08\_agent-01\_learning-curve.png)  
![graph](round-08\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 98.94%     | rootkit\_sanitizer              |
| beurk              | 2.40%      | rootkit\_sanitizer              |
| the\_tick           | 99.80%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 97.18%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 0.24%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 29.94%     | rootkit\_sanitizer              |
| beurk              | 53.46%     | rootkit\_sanitizer              |
| the\_tick           | 99.02%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 74.30%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.02%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 56.95%     | rootkit\_sanitizer              |
| beurk              | 15.13%     | rootkit\_sanitizer              |
| the\_tick           | 99.54%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 89.79%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 41.73%     | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 9/10  
  
- Training Round 9 on Client 1 took 4.07s  
- Training Round 9 on Client 2 took 3.86s  
![graph](round-09\_agent-01\_learning-curve.png)  
![graph](round-09\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 99.11%     | rootkit\_sanitizer              |
| beurk              | 4.59%      | rootkit\_sanitizer              |
| the\_tick           | 99.67%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 95.42%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 0.24%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 27.37%     | rootkit\_sanitizer              |
| beurk              | 62.42%     | rootkit\_sanitizer              |
| the\_tick           | 97.39%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 68.43%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.40%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 60.32%     | rootkit\_sanitizer              |
| beurk              | 20.60%     | rootkit\_sanitizer              |
| the\_tick           | 99.09%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 86.03%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 58.39%     | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 10/10  
  
- Training Round 10 on Client 1 took 5.96s  
- Training Round 10 on Client 2 took 5.02s  
![graph](round-10\_agent-01\_learning-curve.png)  
![graph](round-10\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 99.29%     | rootkit\_sanitizer              |
| beurk              | 7.73%      | rootkit\_sanitizer              |
| the\_tick           | 99.35%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 92.96%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 0.73%      | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 31.27%     | rootkit\_sanitizer              |
| beurk              | 62.63%     | rootkit\_sanitizer              |
| the\_tick           | 97.71%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 69.01%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.85%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 0.00%      | ransomware\_file\_extension\_hide |
| bdvl               | 65.63%     | rootkit\_sanitizer              |
| beurk              | 19.71%     | rootkit\_sanitizer              |
| the\_tick           | 98.82%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 85.92%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 63.50%     | cnc\_ip\_shuffle                 |  

 ### Total training time: 159.66s  
