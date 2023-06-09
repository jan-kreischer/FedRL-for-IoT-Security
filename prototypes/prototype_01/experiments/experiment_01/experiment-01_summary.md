  
# Prototype 1 (Experiment 1)  
---  
  
Executed on 30.03.2023  
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
  
- Training Round 1 on Client 1 took 5.51s  
- Training Round 1 on Client 2 took 5.41s  
![graph](round-01\_agent-01\_learning-curve.png)  
![graph](round-01\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.82 | ransomware\_file\_extension\_hide |
| bdvl               |      98.67 | rootkit\_sanitizer              |
| beurk              |      32.24 | rootkit\_sanitizer              |
| the\_tick           |      86.94 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      72.07 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.56 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |      99.88 | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      96.78 | ransomware\_file\_extension\_hide |
| bdvl               |      98.41 | rootkit\_sanitizer              |
| beurk              |      26.9  | rootkit\_sanitizer              |
| the\_tick           |      94.58 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      78.29 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.56 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |      99.88 | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.77 | ransomware\_file\_extension\_hide |
| bdvl               |      98.32 | rootkit\_sanitizer              |
| beurk              |      18.34 | rootkit\_sanitizer              |
| the\_tick           |      94.38 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      83.69 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 2/10  
  
- Training Round 2 on Client 1 took 4.28s  
- Training Round 2 on Client 2 took 4.1s  
![graph](round-02\_agent-01\_learning-curve.png)  
![graph](round-02\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.82 | ransomware\_file\_extension\_hide |
| bdvl               |      98.49 | rootkit\_sanitizer              |
| beurk              |      69.47 | rootkit\_sanitizer              |
| the\_tick           |      92.49 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      40.14 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.65 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.71 | ransomware\_file\_extension\_hide |
| bdvl               |      98.32 | rootkit\_sanitizer              |
| beurk              |      59.48 | rootkit\_sanitizer              |
| the\_tick           |      97    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      49.65 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.73 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.77 | ransomware\_file\_extension\_hide |
| bdvl               |      98.41 | rootkit\_sanitizer              |
| beurk              |      68.72 | rootkit\_sanitizer              |
| the\_tick           |      94.64 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      42.14 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.73 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 3/10  
  
- Training Round 3 on Client 1 took 3.92s  
- Training Round 3 on Client 2 took 3.85s  
![graph](round-03\_agent-01\_learning-curve.png)  
![graph](round-03\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      97.53 | ransomware\_file\_extension\_hide |
| bdvl               |      98.67 | rootkit\_sanitizer              |
| beurk              |      49.42 | rootkit\_sanitizer              |
| the\_tick           |      96.28 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      69.84 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.29 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.34 | ransomware\_file\_extension\_hide |
| bdvl               |      98.76 | rootkit\_sanitizer              |
| beurk              |      44.28 | rootkit\_sanitizer              |
| the\_tick           |      97.98 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      71.01 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.47 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.02 | ransomware\_file\_extension\_hide |
| bdvl               |      98.76 | rootkit\_sanitizer              |
| beurk              |      39.56 | rootkit\_sanitizer              |
| the\_tick           |      98.3  | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      77.58 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.56 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 4/10  
  
- Training Round 4 on Client 1 took 4.79s  
- Training Round 4 on Client 2 took 3.25s  
![graph](round-04\_agent-01\_learning-curve.png)  
![graph](round-04\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.55 | ransomware\_file\_extension\_hide |
| bdvl               |      98.67 | rootkit\_sanitizer              |
| beurk              |      56.61 | rootkit\_sanitizer              |
| the\_tick           |      97.65 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      67.61 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.56 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.82 | ransomware\_file\_extension\_hide |
| bdvl               |      98.94 | rootkit\_sanitizer              |
| beurk              |      73.1  | rootkit\_sanitizer              |
| the\_tick           |      96.28 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      47.42 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.47 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.77 | ransomware\_file\_extension\_hide |
| bdvl               |      99.03 | rootkit\_sanitizer              |
| beurk              |      70.98 | rootkit\_sanitizer              |
| the\_tick           |      96.34 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      50.35 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.47 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 5/10  
  
- Training Round 5 on Client 1 took 2.97s  
- Training Round 5 on Client 2 took 2.83s  
![graph](round-05\_agent-01\_learning-curve.png)  
![graph](round-05\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.77 | ransomware\_file\_extension\_hide |
| bdvl               |      98.94 | rootkit\_sanitizer              |
| beurk              |      57.63 | rootkit\_sanitizer              |
| the\_tick           |      97.98 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      63.03 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.65 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.55 | ransomware\_file\_extension\_hide |
| bdvl               |      98.58 | rootkit\_sanitizer              |
| beurk              |      74.06 | rootkit\_sanitizer              |
| the\_tick           |      94.51 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      49.06 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      98.85 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.71 | ransomware\_file\_extension\_hide |
| bdvl               |      98.58 | rootkit\_sanitizer              |
| beurk              |      59.69 | rootkit\_sanitizer              |
| the\_tick           |      98.04 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      62.21 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.65 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 6/10  
  
- Training Round 6 on Client 1 took 2.63s  
- Training Round 6 on Client 2 took 2.64s  
![graph](round-06\_agent-01\_learning-curve.png)  
![graph](round-06\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.77 | ransomware\_file\_extension\_hide |
| bdvl               |      99.11 | rootkit\_sanitizer              |
| beurk              |      68.1  | rootkit\_sanitizer              |
| the\_tick           |      96.02 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      59.27 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      98.85 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.82 | ransomware\_file\_extension\_hide |
| bdvl               |      98.67 | rootkit\_sanitizer              |
| beurk              |      52.57 | rootkit\_sanitizer              |
| the\_tick           |      97.84 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      72.54 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.2  | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.71 | ransomware\_file\_extension\_hide |
| bdvl               |      99.11 | rootkit\_sanitizer              |
| beurk              |      64.48 | rootkit\_sanitizer              |
| the\_tick           |      97    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      63.26 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.47 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 7/10  
  
- Training Round 7 on Client 1 took 3695.37s  
- Training Round 7 on Client 2 took 3.59s  
![graph](round-07\_agent-01\_learning-curve.png)  
![graph](round-07\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.77 | ransomware\_file\_extension\_hide |
| bdvl               |      99.11 | rootkit\_sanitizer              |
| beurk              |      72.28 | rootkit\_sanitizer              |
| the\_tick           |      97.58 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      54.81 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.56 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.87 | ransomware\_file\_extension\_hide |
| bdvl               |      99.29 | rootkit\_sanitizer              |
| beurk              |      62.08 | rootkit\_sanitizer              |
| the\_tick           |      97.78 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      64.55 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.47 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.87 | ransomware\_file\_extension\_hide |
| bdvl               |      99.29 | rootkit\_sanitizer              |
| beurk              |      59.41 | rootkit\_sanitizer              |
| the\_tick           |      98.3  | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      68.78 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.56 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 8/10  
  
- Training Round 8 on Client 1 took 9.47s  
- Training Round 8 on Client 2 took 5.01s  
![graph](round-08\_agent-01\_learning-curve.png)  
![graph](round-08\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.66 | ransomware\_file\_extension\_hide |
| bdvl               |      98.58 | rootkit\_sanitizer              |
| beurk              |      59    | rootkit\_sanitizer              |
| the\_tick           |      99.09 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      70.31 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.65 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.77 | ransomware\_file\_extension\_hide |
| bdvl               |      98.94 | rootkit\_sanitizer              |
| beurk              |      55.03 | rootkit\_sanitizer              |
| the\_tick           |      98.3  | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      75    | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.56 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.77 | ransomware\_file\_extension\_hide |
| bdvl               |      99.03 | rootkit\_sanitizer              |
| beurk              |      63.04 | rootkit\_sanitizer              |
| the\_tick           |      98.37 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      68.31 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.56 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 9/10  
  
- Training Round 9 on Client 1 took 3.87s  
- Training Round 9 on Client 2 took 4.38s  
![graph](round-09\_agent-01\_learning-curve.png)  
![graph](round-09\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.82 | ransomware\_file\_extension\_hide |
| bdvl               |      99.2  | rootkit\_sanitizer              |
| beurk              |      54.14 | rootkit\_sanitizer              |
| the\_tick           |      98.3  | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      75.12 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.65 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.98 | ransomware\_file\_extension\_hide |
| bdvl               |      99.38 | rootkit\_sanitizer              |
| beurk              |      62.22 | rootkit\_sanitizer              |
| the\_tick           |      98.24 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      70.19 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.65 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |      99.76 | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.93 | ransomware\_file\_extension\_hide |
| bdvl               |      99.38 | rootkit\_sanitizer              |
| beurk              |      58.18 | rootkit\_sanitizer              |
| the\_tick           |      98.11 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      75.35 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.56 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 10/10  
  
- Training Round 10 on Client 1 took 3.64s  
- Training Round 10 on Client 2 took 3.84s  
![graph](round-10\_agent-01\_learning-curve.png)  
![graph](round-10\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.87 | ransomware\_file\_extension\_hide |
| bdvl               |      98.94 | rootkit\_sanitizer              |
| beurk              |      58.18 | rootkit\_sanitizer              |
| the\_tick           |      98.43 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      72.07 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.65 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.93 | ransomware\_file\_extension\_hide |
| bdvl               |      99.38 | rootkit\_sanitizer              |
| beurk              |      58.59 | rootkit\_sanitizer              |
| the\_tick           |      98.5  | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      72.07 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.65 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      99.14 | ransomware\_file\_extension\_hide |
| bdvl               |      99.2  | rootkit\_sanitizer              |
| beurk              |      78.64 | rootkit\_sanitizer              |
| the\_tick           |      96.8  | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      56.57 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.29 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  

 ### Total training time: 6653.87s  
