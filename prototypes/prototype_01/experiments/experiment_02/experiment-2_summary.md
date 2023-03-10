  
# Prototype 1 (Experiment 2)  
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
- 4799 samples of Behavior.NORMAL  
- 3091 samples of Behavior.RANSOMWARE\_POC  
- 1703 samples of Behavior.ROOTKIT\_BDVL  
- 2372 samples of Behavior.ROOTKIT\_BEURK  
- 2476 samples of Behavior.CNC\_THETICK  
- 1190 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 1828 samples of Behavior.CNC\_OPT1  
- 1272 samples of Behavior.CNC\_OPT2  
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
- 4799 samples of Behavior.NORMAL  
- 3090 samples of Behavior.RANSOMWARE\_POC  
- 1703 samples of Behavior.ROOTKIT\_BDVL  
- 2372 samples of Behavior.ROOTKIT\_BEURK  
- 2476 samples of Behavior.CNC\_THETICK  
- 1190 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 1827 samples of Behavior.CNC\_OPT1  
- 1272 samples of Behavior.CNC\_OPT2  
### Global Agent  
- id: 0  
- batch\_size: 100  
- epsilon: 0  
- batch\_size: 46  
- batch\_size: 4  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 1/10  
  
- Training Round 1 on Client 1 took 6.77s  
- Training Round 1 on Client 2 took 7.12s  
![graph](round-01\_agent-01\_learning-curve.png)  
![graph](round-01\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 97.64%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.23%     | rootkit\_sanitizer              |
| beurk              | 56.61%     | rootkit\_sanitizer              |
| the\_tick           | 88.31%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 52.00%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.82%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 97.91%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.49%     | rootkit\_sanitizer              |
| beurk              | 29.77%     | rootkit\_sanitizer              |
| the\_tick           | 93.66%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 75.94%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 97.43%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.32%     | rootkit\_sanitizer              |
| beurk              | 48.19%     | rootkit\_sanitizer              |
| the\_tick           | 88.57%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 61.62%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 100.00%    | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 2/10  
  
- Training Round 2 on Client 1 took 6.97s  
- Training Round 2 on Client 2 took 8.22s  
![graph](round-02\_agent-01\_learning-curve.png)  
![graph](round-02\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.93%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.58%     | rootkit\_sanitizer              |
| beurk              | 42.71%     | rootkit\_sanitizer              |
| the\_tick           | 95.30%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 70.89%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.98%     | ransomware\_file\_extension\_hide |
| bdvl               | 97.79%     | rootkit\_sanitizer              |
| beurk              | 46.41%     | rootkit\_sanitizer              |
| the\_tick           | 94.77%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 65.85%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.38%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 99.88%     | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.82%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.85%     | rootkit\_sanitizer              |
| beurk              | 65.02%     | rootkit\_sanitizer              |
| the\_tick           | 92.68%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 48.94%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 3/10  
  
- Training Round 3 on Client 1 took 6.19s  
- Training Round 3 on Client 2 took 4.71s  
![graph](round-03\_agent-01\_learning-curve.png)  
![graph](round-03\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.82%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.49%     | rootkit\_sanitizer              |
| beurk              | 57.56%     | rootkit\_sanitizer              |
| the\_tick           | 95.10%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 59.27%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 97.87%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 97.91%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.41%     | rootkit\_sanitizer              |
| beurk              | 42.16%     | rootkit\_sanitizer              |
| the\_tick           | 96.93%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 73.59%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.29%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.87%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.67%     | rootkit\_sanitizer              |
| beurk              | 51.75%     | rootkit\_sanitizer              |
| the\_tick           | 96.54%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 69.01%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.85%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 4/10  
  
- Training Round 4 on Client 1 took 4.93s  
- Training Round 4 on Client 2 took 4.85s  
![graph](round-04\_agent-01\_learning-curve.png)  
![graph](round-04\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.77%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.58%     | rootkit\_sanitizer              |
| beurk              | 61.12%     | rootkit\_sanitizer              |
| the\_tick           | 94.97%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 62.32%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.02%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.98%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.41%     | rootkit\_sanitizer              |
| beurk              | 42.92%     | rootkit\_sanitizer              |
| the\_tick           | 96.80%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 76.29%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.29%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 99.04%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.49%     | rootkit\_sanitizer              |
| beurk              | 51.81%     | rootkit\_sanitizer              |
| the\_tick           | 96.28%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 71.60%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.29%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 5/10  
  
- Training Round 5 on Client 1 took 4.72s  
- Training Round 5 on Client 2 took 5.1s  
![graph](round-05\_agent-01\_learning-curve.png)  
![graph](round-05\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 99.14%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.67%     | rootkit\_sanitizer              |
| beurk              | 67.56%     | rootkit\_sanitizer              |
| the\_tick           | 97.78%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 53.17%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.29%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.82%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.49%     | rootkit\_sanitizer              |
| beurk              | 52.98%     | rootkit\_sanitizer              |
| the\_tick           | 96.08%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 71.83%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.85%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 99.04%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.85%     | rootkit\_sanitizer              |
| beurk              | 59.62%     | rootkit\_sanitizer              |
| the\_tick           | 97.00%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 63.26%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.02%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 6/10  
  
- Training Round 6 on Client 1 took 6.18s  
- Training Round 6 on Client 2 took 5.95s  
![graph](round-06\_agent-01\_learning-curve.png)  
![graph](round-06\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.87%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.94%     | rootkit\_sanitizer              |
| beurk              | 71.73%     | rootkit\_sanitizer              |
| the\_tick           | 97.52%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 51.29%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.02%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.82%     | ransomware\_file\_extension\_hide |
| bdvl               | 94.95%     | rootkit\_sanitizer              |
| beurk              | 55.58%     | rootkit\_sanitizer              |
| the\_tick           | 97.58%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 71.71%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.47%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.82%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.41%     | rootkit\_sanitizer              |
| beurk              | 59.07%     | rootkit\_sanitizer              |
| the\_tick           | 97.98%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 66.55%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 7/10  
  
- Training Round 7 on Client 1 took 4.26s  
- Training Round 7 on Client 2 took 3.67s  
![graph](round-07\_agent-01\_learning-curve.png)  
![graph](round-07\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 99.14%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.85%     | rootkit\_sanitizer              |
| beurk              | 66.94%     | rootkit\_sanitizer              |
| the\_tick           | 97.39%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 62.21%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.38%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.55%     | ransomware\_file\_extension\_hide |
| bdvl               | 95.84%     | rootkit\_sanitizer              |
| beurk              | 47.57%     | rootkit\_sanitizer              |
| the\_tick           | 98.56%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 83.57%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.82%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.58%     | rootkit\_sanitizer              |
| beurk              | 57.84%     | rootkit\_sanitizer              |
| the\_tick           | 97.91%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 73.83%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 8/10  
  
- Training Round 8 on Client 1 took 4.21s  
- Training Round 8 on Client 2 took 4.49s  
![graph](round-08\_agent-01\_learning-curve.png)  
![graph](round-08\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 99.20%     | ransomware\_file\_extension\_hide |
| bdvl               | 99.03%     | rootkit\_sanitizer              |
| beurk              | 65.37%     | rootkit\_sanitizer              |
| the\_tick           | 97.00%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 62.68%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.76%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 99.88%     | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.82%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.67%     | rootkit\_sanitizer              |
| beurk              | 57.22%     | rootkit\_sanitizer              |
| the\_tick           | 97.78%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 70.89%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.98%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.67%     | rootkit\_sanitizer              |
| beurk              | 57.70%     | rootkit\_sanitizer              |
| the\_tick           | 97.84%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 71.95%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.47%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 9/10  
  
- Training Round 9 on Client 1 took 4.69s  
- Training Round 9 on Client 2 took 4.55s  
![graph](round-09\_agent-01\_learning-curve.png)  
![graph](round-09\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.39%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.94%     | rootkit\_sanitizer              |
| beurk              | 66.80%     | rootkit\_sanitizer              |
| the\_tick           | 97.78%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 64.32%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.66%     | ransomware\_file\_extension\_hide |
| bdvl               | 97.79%     | rootkit\_sanitizer              |
| beurk              | 69.47%     | rootkit\_sanitizer              |
| the\_tick           | 97.39%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 60.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 99.04%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.41%     | rootkit\_sanitizer              |
| beurk              | 68.99%     | rootkit\_sanitizer              |
| the\_tick           | 98.17%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 68.19%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 10/10  
  
- Training Round 10 on Client 1 took 4.65s  
- Training Round 10 on Client 2 took 3.54s  
![graph](round-10\_agent-01\_learning-curve.png)  
![graph](round-10\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 99.20%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.58%     | rootkit\_sanitizer              |
| beurk              | 72.07%     | rootkit\_sanitizer              |
| the\_tick           | 96.93%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 53.64%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.02%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 97.48%     | ransomware\_file\_extension\_hide |
| bdvl               | 97.96%     | rootkit\_sanitizer              |
| beurk              | 53.66%     | rootkit\_sanitizer              |
| the\_tick           | 97.91%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 74.53%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 99.88%     | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.98%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.67%     | rootkit\_sanitizer              |
| beurk              | 66.60%     | rootkit\_sanitizer              |
| the\_tick           | 97.91%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 63.62%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.47%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  

 ### Total training time: 134.95s  
