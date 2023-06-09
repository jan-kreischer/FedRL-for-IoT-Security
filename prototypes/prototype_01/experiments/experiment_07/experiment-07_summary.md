  
# Prototype 1 (Experiment 7)  
---  
  
Executed on 06.03.2023  
## Configuration  
### Server  
- nr\_clients: 2  
- nr\_rounds: 100  
- nr\_epochs\_per\_round: 100  
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
### Global Agent  
- id: 0  
- batch\_size: 100  
- epsilon: 0  
- batch\_size: 46  
- batch\_size: 4  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 10/100  
  
- Training Round 10 on Client 1 took 0.65s  
- Training Round 10 on Client 2 took 0.62s  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.45%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.41%     | rootkit\_sanitizer              |
| beurk              | 35.39%     | rootkit\_sanitizer              |
| the\_tick           | 87.33%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 67.84%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.91%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.45%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.67%     | rootkit\_sanitizer              |
| beurk              | 71.66%     | rootkit\_sanitizer              |
| the\_tick           | 75.05%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 32.51%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.11%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.50%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.49%     | rootkit\_sanitizer              |
| beurk              | 52.91%     | rootkit\_sanitizer              |
| the\_tick           | 82.36%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 53.76%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 20/100  
  
- Training Round 20 on Client 1 took 0.77s  
- Training Round 20 on Client 2 took 0.76s  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.71%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.32%     | rootkit\_sanitizer              |
| beurk              | 30.18%     | rootkit\_sanitizer              |
| the\_tick           | 97.00%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 78.87%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.82%     | ransomware\_file\_extension\_hide |
| bdvl               | 97.87%     | rootkit\_sanitizer              |
| beurk              | 50.58%     | rootkit\_sanitizer              |
| the\_tick           | 95.17%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 60.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.77%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.32%     | rootkit\_sanitizer              |
| beurk              | 45.86%     | rootkit\_sanitizer              |
| the\_tick           | 95.23%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 63.73%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 30/100  
  
- Training Round 30 on Client 1 took 0.44s  
- Training Round 30 on Client 2 took 0.46s  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.45%     | ransomware\_file\_extension\_hide |
| bdvl               | 99.38%     | rootkit\_sanitizer              |
| beurk              | 77.41%     | rootkit\_sanitizer              |
| the\_tick           | 92.55%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 35.21%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.05%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 99.15%     | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.61%     | ransomware\_file\_extension\_hide |
| bdvl               | 99.20%     | rootkit\_sanitizer              |
| beurk              | 67.35%     | rootkit\_sanitizer              |
| the\_tick           | 95.43%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 48.83%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.67%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.71%     | ransomware\_file\_extension\_hide |
| bdvl               | 99.03%     | rootkit\_sanitizer              |
| beurk              | 66.19%     | rootkit\_sanitizer              |
| the\_tick           | 94.97%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 50.12%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.49%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 40/100  
  
- Training Round 40 on Client 1 took 0.53s  
- Training Round 40 on Client 2 took 0.55s  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.50%     | ransomware\_file\_extension\_hide |
| bdvl               | 99.03%     | rootkit\_sanitizer              |
| beurk              | 47.43%     | rootkit\_sanitizer              |
| the\_tick           | 96.86%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 71.01%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 97.96%     | ransomware\_file\_extension\_hide |
| bdvl               | 99.11%     | rootkit\_sanitizer              |
| beurk              | 36.00%     | rootkit\_sanitizer              |
| the\_tick           | 98.17%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 78.99%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.61%     | ransomware\_file\_extension\_hide |
| bdvl               | 99.03%     | rootkit\_sanitizer              |
| beurk              | 39.56%     | rootkit\_sanitizer              |
| the\_tick           | 97.78%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 76.64%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 50/100  
  
- Training Round 50 on Client 1 took 0.58s  
- Training Round 50 on Client 2 took 0.5s  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.07%     | ransomware\_file\_extension\_hide |
| bdvl               | 99.11%     | rootkit\_sanitizer              |
| beurk              | 73.65%     | rootkit\_sanitizer              |
| the\_tick           | 94.97%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 47.07%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.76%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.39%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.67%     | rootkit\_sanitizer              |
| beurk              | 44.83%     | rootkit\_sanitizer              |
| the\_tick           | 97.45%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 76.53%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.50%     | ransomware\_file\_extension\_hide |
| bdvl               | 99.03%     | rootkit\_sanitizer              |
| beurk              | 56.81%     | rootkit\_sanitizer              |
| the\_tick           | 96.86%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 66.43%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 60/100  
  
- Training Round 60 on Client 1 took 0.5s  
- Training Round 60 on Client 2 took 0.49s  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.55%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.85%     | rootkit\_sanitizer              |
| beurk              | 52.64%     | rootkit\_sanitizer              |
| the\_tick           | 98.24%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 68.54%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.55%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.94%     | rootkit\_sanitizer              |
| beurk              | 47.78%     | rootkit\_sanitizer              |
| the\_tick           | 98.17%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 73.71%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.66%     | ransomware\_file\_extension\_hide |
| bdvl               | 99.03%     | rootkit\_sanitizer              |
| beurk              | 67.76%     | rootkit\_sanitizer              |
| the\_tick           | 97.26%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 55.87%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.47%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 70/100  
  
- Training Round 70 on Client 1 took 0.55s  
- Training Round 70 on Client 2 took 0.36s  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.77%     | ransomware\_file\_extension\_hide |
| bdvl               | 99.20%     | rootkit\_sanitizer              |
| beurk              | 48.12%     | rootkit\_sanitizer              |
| the\_tick           | 97.26%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 75.12%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.47%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.82%     | ransomware\_file\_extension\_hide |
| bdvl               | 99.11%     | rootkit\_sanitizer              |
| beurk              | 75.98%     | rootkit\_sanitizer              |
| the\_tick           | 95.89%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 49.77%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.38%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.77%     | ransomware\_file\_extension\_hide |
| bdvl               | 99.20%     | rootkit\_sanitizer              |
| beurk              | 70.64%     | rootkit\_sanitizer              |
| the\_tick           | 96.08%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 55.99%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.38%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 99.88%     | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 80/100  
  
- Training Round 80 on Client 1 took 0.46s  
- Training Round 80 on Client 2 took 0.56s  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.82%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.76%     | rootkit\_sanitizer              |
| beurk              | 74.40%     | rootkit\_sanitizer              |
| the\_tick           | 97.06%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 54.58%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.38%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.82%     | ransomware\_file\_extension\_hide |
| bdvl               | 99.03%     | rootkit\_sanitizer              |
| beurk              | 78.71%     | rootkit\_sanitizer              |
| the\_tick           | 95.95%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 42.96%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 98.23%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.82%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.76%     | rootkit\_sanitizer              |
| beurk              | 58.93%     | rootkit\_sanitizer              |
| the\_tick           | 97.78%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 67.96%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.38%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 90/100  
  
- Training Round 90 on Client 1 took 0.34s  
- Training Round 90 on Client 2 took 0.32s  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.71%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.85%     | rootkit\_sanitizer              |
| beurk              | 54.14%     | rootkit\_sanitizer              |
| the\_tick           | 98.17%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 72.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.87%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.76%     | rootkit\_sanitizer              |
| beurk              | 55.58%     | rootkit\_sanitizer              |
| the\_tick           | 98.50%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 72.18%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.82%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.85%     | rootkit\_sanitizer              |
| beurk              | 68.24%     | rootkit\_sanitizer              |
| the\_tick           | 97.78%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 61.62%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 100/100  
  
- Training Round 100 on Client 1 took 0.36s  
- Training Round 100 on Client 2 took 0.3s  


Agent 1
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 98.98%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.85%     | rootkit\_sanitizer              |
| beurk              | 60.37%     | rootkit\_sanitizer              |
| the\_tick           | 96.47%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 70.07%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.47%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 99.09%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.76%     | rootkit\_sanitizer              |
| beurk              | 84.39%     | rootkit\_sanitizer              |
| the\_tick           | 94.84%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 40.49%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.65%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           | Accuracy   | Objective                      |
|:-------------------|:-----------|:-------------------------------|
| ransomware\_poc     | 99.20%     | ransomware\_file\_extension\_hide |
| bdvl               | 98.85%     | rootkit\_sanitizer              |
| beurk              | 83.71%     | rootkit\_sanitizer              |
| the\_tick           | 95.56%     | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar | 46.48%     | cnc\_ip\_shuffle                 |
| data\_leak\_1        | 99.56%     | cnc\_ip\_shuffle                 |
| data\_leak\_2        | 100.00%    | cnc\_ip\_shuffle                 |  

 ### Total training time: 119.12s  
