  
# Prototype 1 (Experiment 1)  
---  
  
Executed on 10.03.2023  
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
- 4952 samples of Behavior.CNC\_THETICK  
- 2380 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3655 samples of Behavior.CNC\_OPT1  
- 2544 samples of Behavior.CNC\_OPT2  
![](behavior\_sample\_distribution\_on\_client-01.png)  
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
- 6181 samples of Behavior.RANSOMWARE\_POC  
- 3406 samples of Behavior.ROOTKIT\_BDVL  
- 4744 samples of Behavior.ROOTKIT\_BEURK  
- 4952 samples of Behavior.CNC\_THETICK  
- 2380 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3655 samples of Behavior.CNC\_OPT1  
- 2544 samples of Behavior.CNC\_OPT2  
![](behavior\_sample\_distribution\_on\_client-02.png)  
### Global Agent  
- id: 0  
- batch\_size: 100  
- epsilon: 0  
- batch\_size: 46  
- batch\_size: 4  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 1/10  
  
- Training Round 1 on Client 1 took 6.24s  
- Training Round 1 on Client 2 took 6.18s  
![graph](round-01\_agent-01\_learning-curve.png)  
![graph](round-01\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.82%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.67%     | rootkit\_sanitizer              |
| beurk              | 32.24%     | rootkit\_sanitizer              |
| the\_tick           | 86.94%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 72.07%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 99.88%     | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 96.78%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.41%     | rootkit\_sanitizer              |
| beurk              | 26.90%     | rootkit\_sanitizer              |
| the\_tick           | 94.58%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 78.29%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 99.88%     | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.77%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.32%     | rootkit\_sanitizer              |
| beurk              | 18.34%     | rootkit\_sanitizer              |
| the\_tick           | 94.38%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 83.69%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 2/10  
  
- Training Round 2 on Client 1 took 7.13s  
- Training Round 2 on Client 2 took 6.55s  
![graph](round-02\_agent-01\_learning-curve.png)  
![graph](round-02\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.82%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.49%     | rootkit\_sanitizer              |
| beurk              | 69.47%     | rootkit\_sanitizer              |
| the\_tick           | 92.49%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 40.14%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.71%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.32%     | rootkit\_sanitizer              |
| beurk              | 59.48%     | rootkit\_sanitizer              |
| the\_tick           | 97.00%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 49.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.77%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.41%     | rootkit\_sanitizer              |
| beurk              | 68.72%     | rootkit\_sanitizer              |
| the\_tick           | 94.64%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 42.14%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 3/10  
  
- Training Round 3 on Client 1 took 6.12s  
- Training Round 3 on Client 2 took 4.11s  
![graph](round-03\_agent-01\_learning-curve.png)  
![graph](round-03\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 97.53%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.67%     | rootkit\_sanitizer              |
| beurk              | 49.42%     | rootkit\_sanitizer              |
| the\_tick           | 96.28%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 69.84%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.29%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.34%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.76%     | rootkit\_sanitizer              |
| beurk              | 44.28%     | rootkit\_sanitizer              |
| the\_tick           | 97.98%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 71.01%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.47%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.02%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.76%     | rootkit\_sanitizer              |
| beurk              | 39.56%     | rootkit\_sanitizer              |
| the\_tick           | 98.30%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 77.58%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 4/10  
  
