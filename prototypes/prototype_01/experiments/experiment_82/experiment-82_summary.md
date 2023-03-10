  
# Prototype 1 (Experiment 82)  
---  
  
Executed on 06.03.2023  
## Configuration  
### Server  
- nr\_clients: 5  
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
- 1920 samples of Behavior.NORMAL  
- 1237 samples of Behavior.RANSOMWARE\_POC  
- 682 samples of Behavior.ROOTKIT\_BDVL  
- 949 samples of Behavior.ROOTKIT\_BEURK  
- 991 samples of Behavior.CNC\_THETICK  
- 476 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 731 samples of Behavior.CNC\_OPT1  
- 509 samples of Behavior.CNC\_OPT2  
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
- 1920 samples of Behavior.NORMAL  
- 1236 samples of Behavior.RANSOMWARE\_POC  
- 681 samples of Behavior.ROOTKIT\_BDVL  
- 949 samples of Behavior.ROOTKIT\_BEURK  
- 991 samples of Behavior.CNC\_THETICK  
- 476 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 731 samples of Behavior.CNC\_OPT1  
- 509 samples of Behavior.CNC\_OPT2  
### Client 3  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.0001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 1920 samples of Behavior.NORMAL  
- 1236 samples of Behavior.RANSOMWARE\_POC  
- 681 samples of Behavior.ROOTKIT\_BDVL  
- 949 samples of Behavior.ROOTKIT\_BEURK  
- 990 samples of Behavior.CNC\_THETICK  
- 476 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 731 samples of Behavior.CNC\_OPT1  
- 509 samples of Behavior.CNC\_OPT2  
### Client 4  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.0001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 1919 samples of Behavior.NORMAL  
- 1236 samples of Behavior.RANSOMWARE\_POC  
- 681 samples of Behavior.ROOTKIT\_BDVL  
- 949 samples of Behavior.ROOTKIT\_BEURK  
- 990 samples of Behavior.CNC\_THETICK  
- 476 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 731 samples of Behavior.CNC\_OPT1  
- 509 samples of Behavior.CNC\_OPT2  
### Client 5  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.0001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 1919 samples of Behavior.NORMAL  
- 1236 samples of Behavior.RANSOMWARE\_POC  
- 681 samples of Behavior.ROOTKIT\_BDVL  
- 948 samples of Behavior.ROOTKIT\_BEURK  
- 990 samples of Behavior.CNC\_THETICK  
- 476 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 731 samples of Behavior.CNC\_OPT1  
- 508 samples of Behavior.CNC\_OPT2  
### Global Agent  
- id: 0  
- batch\_size: 100  
- epsilon: 0  
- batch\_size: 46  
- batch\_size: 4  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 1/10  
  
- Training Round 1 on Client 1 took 11.23s  
- Training Round 1 on Client 2 took 11.72s  
- Training Round 1 on Client 3 took 12.66s  
- Training Round 1 on Client 4 took 15.31s  
- Training Round 1 on Client 5 took 10.82s  
![graph](round-01\_agent-01\_learning-curve.png)  
![graph](round-01\_agent-02\_learning-curve.png)  
![graph](round-01\_agent-03\_learning-curve.png)  
![graph](round-01\_agent-04\_learning-curve.png)  
![graph](round-01\_agent-05\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.07%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.14%     | rootkit\_sanitizer              |
| beurk              | 36.69%     | rootkit\_sanitizer              |
| the\_tick           | 94.06%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 68.08%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.50%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.76%     | rootkit\_sanitizer              |
| beurk              | 43.05%     | rootkit\_sanitizer              |
| the\_tick           | 82.43%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 61.62%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.85%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 99.39%     | cnc\_ip\_shuffle                 |  


Agent 3
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.23%     | ransomware\_file\_extension\_hide |
| bdvl               | 95.84%     | rootkit\_sanitizer              |
| beurk              | 35.32%     | rootkit\_sanitizer              |
| the\_tick           | 92.75%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 78.64%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.02%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 4
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.61%     | ransomware\_file\_extension\_hide |
| bdvl               | 96.19%     | rootkit\_sanitizer              |
| beurk              | 23.13%     | rootkit\_sanitizer              |
| the\_tick           | 94.71%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 84.04%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 5
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.12%     | ransomware\_file\_extension\_hide |
| bdvl               | 94.33%     | rootkit\_sanitizer              |
| beurk              | 11.16%     | rootkit\_sanitizer              |
| the\_tick           | 96.73%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 90.38%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.91%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 4.02%      | ransomware\_file\_extension\_hide |
| bdvl               | 98.14%     | rootkit\_sanitizer              |
| beurk              | 25.33%     | rootkit\_sanitizer              |
| the\_tick           | 93.27%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 79.34%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.82%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 2/10  
  
- Training Round 2 on Client 1 took 10.73s  
- Training Round 2 on Client 2 took 8.28s  
- Training Round 2 on Client 3 took 8.73s  
- Training Round 2 on Client 4 took 8.63s  
- Training Round 2 on Client 5 took 8.47s  
![graph](round-02\_agent-01\_learning-curve.png)  
![graph](round-02\_agent-02\_learning-curve.png)  
![graph](round-02\_agent-03\_learning-curve.png)  
![graph](round-02\_agent-04\_learning-curve.png)  
![graph](round-02\_agent-05\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.45%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.67%     | rootkit\_sanitizer              |
| beurk              | 69.61%     | rootkit\_sanitizer              |
| the\_tick           | 90.73%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 46.36%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.76%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.71%     | ransomware\_file\_extension\_hide |
| bdvl               | 97.52%     | rootkit\_sanitizer              |
| beurk              | 51.88%     | rootkit\_sanitizer              |
| the\_tick           | 94.84%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 61.62%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 3
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.39%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.41%     | rootkit\_sanitizer              |
| beurk              | 36.34%     | rootkit\_sanitizer              |
| the\_tick           | 95.89%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 74.88%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.82%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 4
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.71%     | ransomware\_file\_extension\_hide |
| bdvl               | 97.08%     | rootkit\_sanitizer              |
| beurk              | 33.26%     | rootkit\_sanitizer              |
| the\_tick           | 96.93%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 80.16%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 5
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.45%     | ransomware\_file\_extension\_hide |
| bdvl               | 97.87%     | rootkit\_sanitizer              |
| beurk              | 42.23%     | rootkit\_sanitizer              |
| the\_tick           | 97.06%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 72.54%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.66%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.67%     | rootkit\_sanitizer              |
| beurk              | 42.71%     | rootkit\_sanitizer              |
| the\_tick           | 96.54%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 72.18%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 3/10  
  
- Training Round 3 on Client 1 took 6.98s  
- Training Round 3 on Client 2 took 6.88s  
- Training Round 3 on Client 3 took 9.1s  
- Training Round 3 on Client 4 took 9.44s  
- Training Round 3 on Client 5 took 8.94s  
![graph](round-03\_agent-01\_learning-curve.png)  
![graph](round-03\_agent-02\_learning-curve.png)  
![graph](round-03\_agent-03\_learning-curve.png)  
![graph](round-03\_agent-04\_learning-curve.png)  
![graph](round-03\_agent-05\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.45%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.76%     | rootkit\_sanitizer              |
| beurk              | 71.12%     | rootkit\_sanitizer              |
| the\_tick           | 92.36%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 44.60%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.40%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.07%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.49%     | rootkit\_sanitizer              |
| beurk              | 69.61%     | rootkit\_sanitizer              |
| the\_tick           | 93.53%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 49.53%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.58%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 3
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.12%     | ransomware\_file\_extension\_hide |
| bdvl               | 97.70%     | rootkit\_sanitizer              |
| beurk              | 47.91%     | rootkit\_sanitizer              |
| the\_tick           | 95.95%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 70.42%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 96.45%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 4
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 97.96%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.14%     | rootkit\_sanitizer              |
| beurk              | 47.98%     | rootkit\_sanitizer              |
| the\_tick           | 94.77%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 69.13%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 5
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.39%     | ransomware\_file\_extension\_hide |
| bdvl               | 96.81%     | rootkit\_sanitizer              |
| beurk              | 27.86%     | rootkit\_sanitizer              |
| the\_tick           | 97.06%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 84.15%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.11%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.18%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.76%     | rootkit\_sanitizer              |
| beurk              | 53.32%     | rootkit\_sanitizer              |
| the\_tick           | 95.62%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 68.43%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.02%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 4/10  
  
- Training Round 4 on Client 1 took 10.3s  
- Training Round 4 on Client 2 took 9.25s  
- Training Round 4 on Client 3 took 8.3s  
- Training Round 4 on Client 4 took 6.13s  
- Training Round 4 on Client 5 took 6.06s  
![graph](round-04\_agent-01\_learning-curve.png)  
![graph](round-04\_agent-02\_learning-curve.png)  
![graph](round-04\_agent-03\_learning-curve.png)  
![graph](round-04\_agent-04\_learning-curve.png)  
![graph](round-04\_agent-05\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.61%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.85%     | rootkit\_sanitizer              |
| beurk              | 74.61%     | rootkit\_sanitizer              |
| the\_tick           | 94.64%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 40.38%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.67%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 99.88%     | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.34%     | ransomware\_file\_extension\_hide |
| bdvl               | 99.20%     | rootkit\_sanitizer              |
| beurk              | 62.90%     | rootkit\_sanitizer              |
| the\_tick           | 96.08%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 61.27%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.05%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 3
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 97.48%     | ransomware\_file\_extension\_hide |
| bdvl               | 94.42%     | rootkit\_sanitizer              |
| beurk              | 72.55%     | rootkit\_sanitizer              |
| the\_tick           | 94.51%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 51.06%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 97.69%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 4
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 97.96%     | ransomware\_file\_extension\_hide |
| bdvl               | 96.90%     | rootkit\_sanitizer              |
| beurk              | 48.25%     | rootkit\_sanitizer              |
| the\_tick           | 96.73%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 73.94%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 5
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.29%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.94%     | rootkit\_sanitizer              |
| beurk              | 60.10%     | rootkit\_sanitizer              |
| the\_tick           | 96.60%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 59.86%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.38%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.66%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.76%     | rootkit\_sanitizer              |
| beurk              | 59.48%     | rootkit\_sanitizer              |
| the\_tick           | 96.80%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 65.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.47%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 5/10  
  
- Training Round 5 on Client 1 took 3.96s  
- Training Round 5 on Client 2 took 3.82s  
- Training Round 5 on Client 3 took 3.98s  
- Training Round 5 on Client 4 took 4.86s  
- Training Round 5 on Client 5 took 5.52s  
![graph](round-05\_agent-01\_learning-curve.png)  
![graph](round-05\_agent-02\_learning-curve.png)  
![graph](round-05\_agent-03\_learning-curve.png)  
![graph](round-05\_agent-04\_learning-curve.png)  
![graph](round-05\_agent-05\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.55%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.67%     | rootkit\_sanitizer              |
| beurk              | 70.57%     | rootkit\_sanitizer              |
| the\_tick           | 96.93%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 52.82%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.94%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.77%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.14%     | rootkit\_sanitizer              |
| beurk              | 67.56%     | rootkit\_sanitizer              |
| the\_tick           | 96.73%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 56.57%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 97.96%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 3
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.18%     | ransomware\_file\_extension\_hide |
| bdvl               | 96.19%     | rootkit\_sanitizer              |
| beurk              | 60.44%     | rootkit\_sanitizer              |
| the\_tick           | 97.06%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 66.55%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.02%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 4
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.18%     | ransomware\_file\_extension\_hide |
| bdvl               | 96.90%     | rootkit\_sanitizer              |
| beurk              | 43.19%     | rootkit\_sanitizer              |
| the\_tick           | 97.91%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 76.64%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 5
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.50%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.14%     | rootkit\_sanitizer              |
| beurk              | 59.89%     | rootkit\_sanitizer              |
| the\_tick           | 96.80%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 62.21%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.47%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.55%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.49%     | rootkit\_sanitizer              |
| beurk              | 61.81%     | rootkit\_sanitizer              |
| the\_tick           | 97.71%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 63.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 6/10  
  
- Training Round 6 on Client 1 took 4.59s  
- Training Round 6 on Client 2 took 5.19s  
- Training Round 6 on Client 3 took 4.48s  
- Training Round 6 on Client 4 took 4.2s  
- Training Round 6 on Client 5 took 4.03s  
![graph](round-06\_agent-01\_learning-curve.png)  
![graph](round-06\_agent-02\_learning-curve.png)  
![graph](round-06\_agent-03\_learning-curve.png)  
![graph](round-06\_agent-04\_learning-curve.png)  
![graph](round-06\_agent-05\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.55%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.85%     | rootkit\_sanitizer              |
| beurk              | 85.69%     | rootkit\_sanitizer              |
| the\_tick           | 93.47%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 30.63%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.85%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.61%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.94%     | rootkit\_sanitizer              |
| beurk              | 67.62%     | rootkit\_sanitizer              |
| the\_tick           | 96.34%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 58.10%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.14%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 3
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.07%     | ransomware\_file\_extension\_hide |
| bdvl               | 97.08%     | rootkit\_sanitizer              |
| beurk              | 71.53%     | rootkit\_sanitizer              |
| the\_tick           | 94.97%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 55.05%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.76%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 4
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.34%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.23%     | rootkit\_sanitizer              |
| beurk              | 62.15%     | rootkit\_sanitizer              |
| the\_tick           | 96.67%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 65.26%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 5
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.61%     | ransomware\_file\_extension\_hide |
| bdvl               | 95.66%     | rootkit\_sanitizer              |
| beurk              | 43.33%     | rootkit\_sanitizer              |
| the\_tick           | 98.37%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 80.87%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.55%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.41%     | rootkit\_sanitizer              |
| beurk              | 69.68%     | rootkit\_sanitizer              |
| the\_tick           | 96.47%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 61.03%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.29%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 7/10  
  
- Training Round 7 on Client 1 took 4.35s  
- Training Round 7 on Client 2 took 4.67s  
- Training Round 7 on Client 3 took 5.74s  
- Training Round 7 on Client 4 took 6.13s  
- Training Round 7 on Client 5 took 5.56s  
![graph](round-07\_agent-01\_learning-curve.png)  
![graph](round-07\_agent-02\_learning-curve.png)  
![graph](round-07\_agent-03\_learning-curve.png)  
![graph](round-07\_agent-04\_learning-curve.png)  
![graph](round-07\_agent-05\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 99.04%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.67%     | rootkit\_sanitizer              |
| beurk              | 73.92%     | rootkit\_sanitizer              |
| the\_tick           | 96.86%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 57.04%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.71%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.76%     | rootkit\_sanitizer              |
| beurk              | 79.47%     | rootkit\_sanitizer              |
| the\_tick           | 94.97%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 45.19%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.02%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 3
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.45%     | ransomware\_file\_extension\_hide |
| bdvl               | 95.31%     | rootkit\_sanitizer              |
| beurk              | 48.87%     | rootkit\_sanitizer              |
| the\_tick           | 98.43%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 81.46%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 4
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.82%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.85%     | rootkit\_sanitizer              |
| beurk              | 60.51%     | rootkit\_sanitizer              |
| the\_tick           | 97.91%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 69.25%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 5
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.55%     | ransomware\_file\_extension\_hide |
| bdvl               | 96.63%     | rootkit\_sanitizer              |
| beurk              | 41.96%     | rootkit\_sanitizer              |
| the\_tick           | 97.91%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 84.62%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.47%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.66%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.76%     | rootkit\_sanitizer              |
| beurk              | 71.39%     | rootkit\_sanitizer              |
| the\_tick           | 96.73%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 60.92%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.47%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 8/10  
  
- Training Round 8 on Client 1 took 6.43s  
- Training Round 8 on Client 2 took 6.15s  
- Training Round 8 on Client 3 took 6.43s  
- Training Round 8 on Client 4 took 6.94s  
- Training Round 8 on Client 5 took 5.53s  
![graph](round-08\_agent-01\_learning-curve.png)  
![graph](round-08\_agent-02\_learning-curve.png)  
![graph](round-08\_agent-03\_learning-curve.png)  
![graph](round-08\_agent-04\_learning-curve.png)  
![graph](round-08\_agent-05\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.12%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.85%     | rootkit\_sanitizer              |
| beurk              | 85.08%     | rootkit\_sanitizer              |
| the\_tick           | 93.60%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 42.84%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.11%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.61%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.94%     | rootkit\_sanitizer              |
| beurk              | 46.82%     | rootkit\_sanitizer              |
| the\_tick           | 97.91%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 78.64%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.47%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 3
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.45%     | ransomware\_file\_extension\_hide |
| bdvl               | 97.08%     | rootkit\_sanitizer              |
| beurk              | 57.22%     | rootkit\_sanitizer              |
| the\_tick           | 97.19%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 75.23%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 4
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.50%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.49%     | rootkit\_sanitizer              |
| beurk              | 42.23%     | rootkit\_sanitizer              |
| the\_tick           | 98.56%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 78.52%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 5
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.66%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.05%     | rootkit\_sanitizer              |
| beurk              | 48.25%     | rootkit\_sanitizer              |
| the\_tick           | 97.98%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 78.64%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.77%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.85%     | rootkit\_sanitizer              |
| beurk              | 66.87%     | rootkit\_sanitizer              |
| the\_tick           | 97.45%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 69.01%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 9/10  
  
- Training Round 9 on Client 1 took 5.11s  
- Training Round 9 on Client 2 took 6.21s  
- Training Round 9 on Client 3 took 5.78s  
- Training Round 9 on Client 4 took 6.71s  
- Training Round 9 on Client 5 took 5.69s  
![graph](round-09\_agent-01\_learning-curve.png)  
![graph](round-09\_agent-02\_learning-curve.png)  
![graph](round-09\_agent-03\_learning-curve.png)  
![graph](round-09\_agent-04\_learning-curve.png)  
![graph](round-09\_agent-05\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.77%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.58%     | rootkit\_sanitizer              |
| beurk              | 86.11%     | rootkit\_sanitizer              |
| the\_tick           | 93.86%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 27.35%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.38%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.82%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.76%     | rootkit\_sanitizer              |
| beurk              | 59.82%     | rootkit\_sanitizer              |
| the\_tick           | 97.65%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 71.13%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 3
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.61%     | ransomware\_file\_extension\_hide |
| bdvl               | 97.79%     | rootkit\_sanitizer              |
| beurk              | 51.33%     | rootkit\_sanitizer              |
| the\_tick           | 98.82%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 80.63%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 4
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.61%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.49%     | rootkit\_sanitizer              |
| beurk              | 62.56%     | rootkit\_sanitizer              |
| the\_tick           | 95.89%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 67.96%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.38%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 5
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 99.14%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.67%     | rootkit\_sanitizer              |
| beurk              | 50.92%     | rootkit\_sanitizer              |
| the\_tick           | 98.56%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 77.23%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.71%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.67%     | rootkit\_sanitizer              |
| beurk              | 67.76%     | rootkit\_sanitizer              |
| the\_tick           | 97.84%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 67.84%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 10/10  
  
- Training Round 10 on Client 1 took 6.77s  
- Training Round 10 on Client 2 took 6.19s  
- Training Round 10 on Client 3 took 5.62s  
- Training Round 10 on Client 4 took 6.82s  
- Training Round 10 on Client 5 took 6.14s  
![graph](round-10\_agent-01\_learning-curve.png)  
![graph](round-10\_agent-02\_learning-curve.png)  
![graph](round-10\_agent-03\_learning-curve.png)  
![graph](round-10\_agent-04\_learning-curve.png)  
![graph](round-10\_agent-05\_learning-curve.png)  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.61%     | ransomware\_file\_extension\_hide |
| bdvl               | 99.38%     | rootkit\_sanitizer              |
| beurk              | 82.14%     | rootkit\_sanitizer              |
| the\_tick           | 95.30%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 42.96%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.38%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 99.03%     | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.61%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.76%     | rootkit\_sanitizer              |
| beurk              | 58.18%     | rootkit\_sanitizer              |
| the\_tick           | 97.91%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 71.71%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 3
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.77%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.58%     | rootkit\_sanitizer              |
| beurk              | 69.47%     | rootkit\_sanitizer              |
| the\_tick           | 96.60%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 68.54%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 4
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.55%     | ransomware\_file\_extension\_hide |
| bdvl               | 97.52%     | rootkit\_sanitizer              |
| beurk              | 71.32%     | rootkit\_sanitizer              |
| the\_tick           | 96.15%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 65.49%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 5
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.71%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.41%     | rootkit\_sanitizer              |
| beurk              | 47.84%     | rootkit\_sanitizer              |
| the\_tick           | 98.89%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 81.81%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.87%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.76%     | rootkit\_sanitizer              |
| beurk              | 67.76%     | rootkit\_sanitizer              |
| the\_tick           | 97.78%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 68.78%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  

 ### Total training time: 438.76s  
