  
# Prototype 1 (Experiment 1)  
---  
  
Executed on 16.04.2023  
## Configuration  
### Server  
- nr\_clients: 2  
- nr\_rounds: 10  
- nr\_epochs\_per\_round: 100  
  
### Client 1  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 9631 samples of Behavior.NORMAL  
- 6195 samples of Behavior.RANSOMWARE\_POC  
- 3437 samples of Behavior.ROOTKIT\_BDVL  
- 4753 samples of Behavior.ROOTKIT\_BEURK  
- 4922 samples of Behavior.CNC\_THETICK  
- 2368 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3632 samples of Behavior.CNC\_OPT1  
- 2532 samples of Behavior.CNC\_OPT2  
### Client 2  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 9631 samples of Behavior.NORMAL  
- 6195 samples of Behavior.RANSOMWARE\_POC  
- 3437 samples of Behavior.ROOTKIT\_BDVL  
- 4753 samples of Behavior.ROOTKIT\_BEURK  
- 4922 samples of Behavior.CNC\_THETICK  
- 2368 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3632 samples of Behavior.CNC\_OPT1  
- 2532 samples of Behavior.CNC\_OPT2  
### Global Agent  
- id: 0  
- batch\_size: 100  
- epsilon: 0  
- batch\_size: 46  
- batch\_size: 4  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 1/10  
  
- Training Round 1 on Client 1 took 4.06s  
- Training Round 1 on Client 2 took 12.08s  
![graph](round-01\_agent-01\_learning-curve.png)  
![graph](round-01\_agent-02\_learning-curve.png)  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 2/10  
  
- Training Round 2 on Client 1 took 4.64s  
- Training Round 2 on Client 2 took 4.5s  
![graph](round-02\_agent-01\_learning-curve.png)  
![graph](round-02\_agent-02\_learning-curve.png)  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 3/10  
  
- Training Round 3 on Client 1 took 4.52s  
- Training Round 3 on Client 2 took 4.14s  
![graph](round-03\_agent-01\_learning-curve.png)  
![graph](round-03\_agent-02\_learning-curve.png)  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 4/10  
  
- Training Round 4 on Client 1 took 4.47s  
- Training Round 4 on Client 2 took 5.66s  
![graph](round-04\_agent-01\_learning-curve.png)  
![graph](round-04\_agent-02\_learning-curve.png)  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 5/10  
  
  
# Prototype 1 (Experiment 1)  
---  
  
Executed on 25.04.2023  
## Configuration  
### Server  
- nr\_clients: 2  
- nr\_rounds: 10  
- nr\_epochs\_per\_round: 100  
  
### Client 1  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 9639 samples of Behavior.NORMAL  
- 6207 samples of Behavior.RANSOMWARE\_POC  
- 3373 samples of Behavior.ROOTKIT\_BDVL  
- 4703 samples of Behavior.ROOTKIT\_BEURK  
- 4949 samples of Behavior.CNC\_THETICK  
- 2378 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3647 samples of Behavior.CNC\_OPT1  
- 2531 samples of Behavior.CNC\_OPT2  
### Client 2  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 9639 samples of Behavior.NORMAL  
- 6207 samples of Behavior.RANSOMWARE\_POC  
- 3373 samples of Behavior.ROOTKIT\_BDVL  
- 4703 samples of Behavior.ROOTKIT\_BEURK  
- 4949 samples of Behavior.CNC\_THETICK  
- 2378 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3647 samples of Behavior.CNC\_OPT1  
- 2531 samples of Behavior.CNC\_OPT2  
### Global Agent  
- id: 0  
- batch\_size: 100  
- epsilon: 0  
- batch\_size: 46  
- batch\_size: 4  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 1/10  
  
- Training Round 1 on Client 1 took 2.9s  
- Training Round 1 on Client 2 took 3.75s  
![graph](round-01\_agent-01\_learning-curve.png)  
![graph](round-01\_agent-02\_learning-curve.png)  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 2/10  
  
- Training Round 2 on Client 1 took 3.45s  
- Training Round 2 on Client 2 took 3.44s  
![graph](round-02\_agent-01\_learning-curve.png)  
![graph](round-02\_agent-02\_learning-curve.png)  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 3/10  
  
- Training Round 3 on Client 1 took 3.65s  
- Training Round 3 on Client 2 took 4.31s  
![graph](round-03\_agent-01\_learning-curve.png)  
![graph](round-03\_agent-02\_learning-curve.png)  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 4/10  
  
- Training Round 4 on Client 1 took 3.17s  
- Training Round 4 on Client 2 took 3.56s  
![graph](round-04\_agent-01\_learning-curve.png)  
![graph](round-04\_agent-02\_learning-curve.png)  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 5/10  
  
- Training Round 5 on Client 1 took 3.76s  
- Training Round 5 on Client 2 took 3.45s  
![graph](round-05\_agent-01\_learning-curve.png)  
![graph](round-05\_agent-02\_learning-curve.png)  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 6/10  
  
- Training Round 6 on Client 1 took 3.87s  
- Training Round 6 on Client 2 took 3.71s  
![graph](round-06\_agent-01\_learning-curve.png)  
![graph](round-06\_agent-02\_learning-curve.png)  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 7/10  
  
- Training Round 7 on Client 1 took 3.01s  
- Training Round 7 on Client 2 took 2.81s  
![graph](round-07\_agent-01\_learning-curve.png)  
![graph](round-07\_agent-02\_learning-curve.png)  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 8/10  
  
- Training Round 8 on Client 1 took 3.28s  
- Training Round 8 on Client 2 took 3.64s  
![graph](round-08\_agent-01\_learning-curve.png)  
![graph](round-08\_agent-02\_learning-curve.png)  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 9/10  
  
- Training Round 9 on Client 1 took 3.92s  
- Training Round 9 on Client 2 took 3.33s  
![graph](round-09\_agent-01\_learning-curve.png)  
![graph](round-09\_agent-02\_learning-curve.png)  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 10/10  
  
- Training Round 10 on Client 1 took 3.44s  
- Training Round 10 on Client 2 took 4.09s  
![graph](round-10\_agent-01\_learning-curve.png)  
![graph](round-10\_agent-02\_learning-curve.png)  
  
# Prototype 1 (Experiment 1)  
---  
  
Executed on 25.04.2023  
## Configuration  
### Server  
- nr\_clients: 2  
- nr\_rounds: 10  
- nr\_epochs\_per\_round: 100  
  
### Client 1  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 9502 samples of Behavior.NORMAL  
- 6212 samples of Behavior.RANSOMWARE\_POC  
- 3414 samples of Behavior.ROOTKIT\_BDVL  
- 4764 samples of Behavior.ROOTKIT\_BEURK  
- 4891 samples of Behavior.CNC\_THETICK  
- 2326 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3609 samples of Behavior.CNC\_OPT1  
- 2535 samples of Behavior.CNC\_OPT2  
### Client 2  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 9502 samples of Behavior.NORMAL  
- 6212 samples of Behavior.RANSOMWARE\_POC  
- 3414 samples of Behavior.ROOTKIT\_BDVL  
- 4764 samples of Behavior.ROOTKIT\_BEURK  
- 4891 samples of Behavior.CNC\_THETICK  
- 2326 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3609 samples of Behavior.CNC\_OPT1  
- 2535 samples of Behavior.CNC\_OPT2  
### Global Agent  
- id: 0  
- batch\_size: 100  
- epsilon: 0  
- batch\_size: 46  
- batch\_size: 4  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 10/10  
  
- Training Round 10 on Client 1 took 1.67s  
- Training Round 10 on Client 2 took 1.77s  
![graph](round-10\_agent-01\_learning-curve.png)  
![graph](round-10\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.29 | ransomware\_file\_extension\_hide |
| bdvl               |      89.73 | rootkit\_sanitizer              |
| beurk              |       9.31 | rootkit\_sanitizer              |
| the\_tick           |      90.07 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      89.55 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.65 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.12 | ransomware\_file\_extension\_hide |
| bdvl               |      99.47 | rootkit\_sanitizer              |
| beurk              |      29.64 | rootkit\_sanitizer              |
| the\_tick           |      69.24 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      74.3  | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      78.97 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |      88.81 | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.34 | ransomware\_file\_extension\_hide |
| bdvl               |      97.87 | rootkit\_sanitizer              |
| beurk              |      25.26 | rootkit\_sanitizer              |
| the\_tick           |      76.49 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      78.29 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      85.63 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |      98.42 | cnc\_ip\_shuffle                 |  
  
# Prototype 1 (Experiment 1)  
---  
  
Executed on 25.04.2023  
## Configuration  
### Server  
- nr\_clients: 2  
- nr\_rounds: 10  
- nr\_epochs\_per\_round: 100  
  
### Client 1  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 9611 samples of Behavior.NORMAL  
- 6207 samples of Behavior.RANSOMWARE\_POC  
- 3413 samples of Behavior.ROOTKIT\_BDVL  
- 4743 samples of Behavior.ROOTKIT\_BEURK  
- 4944 samples of Behavior.CNC\_THETICK  
- 2364 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3624 samples of Behavior.CNC\_OPT1  
- 2550 samples of Behavior.CNC\_OPT2  
### Client 2  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 9611 samples of Behavior.NORMAL  
- 6207 samples of Behavior.RANSOMWARE\_POC  
- 3413 samples of Behavior.ROOTKIT\_BDVL  
- 4743 samples of Behavior.ROOTKIT\_BEURK  
- 4944 samples of Behavior.CNC\_THETICK  
- 2364 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3624 samples of Behavior.CNC\_OPT1  
- 2550 samples of Behavior.CNC\_OPT2  
### Global Agent  
- id: 0  
- batch\_size: 100  
- epsilon: 0  
- batch\_size: 46  
- batch\_size: 4  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 10/10  
  
- Training Round 10 on Client 1 took 5.29s  
- Training Round 10 on Client 2 took 4.45s  
![graph](round-10\_agent-01\_learning-curve.png)  
![graph](round-10\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.55 | ransomware\_file\_extension\_hide |
| bdvl               |      98.94 | rootkit\_sanitizer              |
| beurk              |      69.68 | rootkit\_sanitizer              |
| the\_tick           |      26.19 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      32.39 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      99.09 | ransomware\_file\_extension\_hide |
| bdvl               |      98.94 | rootkit\_sanitizer              |
| beurk              |      74.61 | rootkit\_sanitizer              |
| the\_tick           |      22.21 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      27.11 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.91 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.87 | ransomware\_file\_extension\_hide |
| bdvl               |      98.94 | rootkit\_sanitizer              |
| beurk              |      71.8  | rootkit\_sanitizer              |
| the\_tick           |      24.04 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      29.46 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
  
# Prototype 1 (Experiment 1)  
---  
  
Executed on 01.05.2023  
## Configuration  
### Server  
- nr\_clients: 10  
- nr\_rounds: 10  
- nr\_epochs\_per\_round: 100  
  
### Client 1  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 4832 samples of Behavior.NORMAL  
- 6185 samples of Behavior.RANSOMWARE\_POC  
- 3442 samples of Behavior.ROOTKIT\_BDVL  
- 4760 samples of Behavior.ROOTKIT\_BEURK  
- 4950 samples of Behavior.CNC\_THETICK  
- 2423 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3637 samples of Behavior.CNC\_OPT1  
- 2530 samples of Behavior.CNC\_OPT2  
### Client 2  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 4832 samples of Behavior.NORMAL  
- 6185 samples of Behavior.RANSOMWARE\_POC  
- 3442 samples of Behavior.ROOTKIT\_BDVL  
- 4760 samples of Behavior.ROOTKIT\_BEURK  
- 4950 samples of Behavior.CNC\_THETICK  
- 2423 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3637 samples of Behavior.CNC\_OPT1  
- 2530 samples of Behavior.CNC\_OPT2  
### Client 3  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 4832 samples of Behavior.NORMAL  
- 6185 samples of Behavior.RANSOMWARE\_POC  
- 3442 samples of Behavior.ROOTKIT\_BDVL  
- 4760 samples of Behavior.ROOTKIT\_BEURK  
- 4950 samples of Behavior.CNC\_THETICK  
- 2423 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3637 samples of Behavior.CNC\_OPT1  
- 2530 samples of Behavior.CNC\_OPT2  
### Client 4  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 4832 samples of Behavior.NORMAL  
- 6185 samples of Behavior.RANSOMWARE\_POC  
- 3442 samples of Behavior.ROOTKIT\_BDVL  
- 4760 samples of Behavior.ROOTKIT\_BEURK  
- 4950 samples of Behavior.CNC\_THETICK  
- 2423 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3637 samples of Behavior.CNC\_OPT1  
- 2530 samples of Behavior.CNC\_OPT2  
### Client 5  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 4832 samples of Behavior.NORMAL  
- 6185 samples of Behavior.RANSOMWARE\_POC  
- 3442 samples of Behavior.ROOTKIT\_BDVL  
- 4760 samples of Behavior.ROOTKIT\_BEURK  
- 4950 samples of Behavior.CNC\_THETICK  
- 2423 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3637 samples of Behavior.CNC\_OPT1  
- 2530 samples of Behavior.CNC\_OPT2  
### Client 6  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 4832 samples of Behavior.NORMAL  
- 6185 samples of Behavior.RANSOMWARE\_POC  
- 3442 samples of Behavior.ROOTKIT\_BDVL  
- 4760 samples of Behavior.ROOTKIT\_BEURK  
- 4950 samples of Behavior.CNC\_THETICK  
- 2423 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3637 samples of Behavior.CNC\_OPT1  
- 2530 samples of Behavior.CNC\_OPT2  
### Client 7  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 4832 samples of Behavior.NORMAL  
- 6185 samples of Behavior.RANSOMWARE\_POC  
- 3442 samples of Behavior.ROOTKIT\_BDVL  
- 4760 samples of Behavior.ROOTKIT\_BEURK  
- 4950 samples of Behavior.CNC\_THETICK  
- 2423 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3637 samples of Behavior.CNC\_OPT1  
- 2530 samples of Behavior.CNC\_OPT2  
### Client 8  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 4832 samples of Behavior.NORMAL  
- 6185 samples of Behavior.RANSOMWARE\_POC  
- 3442 samples of Behavior.ROOTKIT\_BDVL  
- 4760 samples of Behavior.ROOTKIT\_BEURK  
- 4950 samples of Behavior.CNC\_THETICK  
- 2423 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3637 samples of Behavior.CNC\_OPT1  
- 2530 samples of Behavior.CNC\_OPT2  
### Client 9  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 4832 samples of Behavior.NORMAL  
- 6185 samples of Behavior.RANSOMWARE\_POC  
- 3442 samples of Behavior.ROOTKIT\_BDVL  
- 4760 samples of Behavior.ROOTKIT\_BEURK  
- 4950 samples of Behavior.CNC\_THETICK  
- 2423 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3637 samples of Behavior.CNC\_OPT1  
- 2530 samples of Behavior.CNC\_OPT2  
### Client 10  
- gamma: 0.1  
- learning\_rate: 0.0001  
- batch\_size: 100  
- epsilon\_max: 1.0  
- epsilon\_min: 0.01  
- epsilon\_decay: 0.001  
- input\_dims: 46  
- output\_dims: 4  
  
Training Data Split  
- 4832 samples of Behavior.NORMAL  
- 6185 samples of Behavior.RANSOMWARE\_POC  
- 3442 samples of Behavior.ROOTKIT\_BDVL  
- 4760 samples of Behavior.ROOTKIT\_BEURK  
- 4950 samples of Behavior.CNC\_THETICK  
- 2423 samples of Behavior.CNC\_BACKDOOR\_JAKORITAR  
- 3637 samples of Behavior.CNC\_OPT1  
- 2530 samples of Behavior.CNC\_OPT2  
### Global Agent  
- id: 0  
- batch\_size: 100  
- epsilon: 0  
- batch\_size: 46  
- batch\_size: 4  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 1/10  
  
- Training Round 1 on Client 1 took 3.42s  
- Training Round 1 on Client 2 took 3.18s  
- Training Round 1 on Client 3 took 3.07s  
- Training Round 1 on Client 4 took 7.07s  
- Training Round 1 on Client 5 took 3.55s  
- Training Round 1 on Client 6 took 3.88s  
- Training Round 1 on Client 7 took 3.22s  
- Training Round 1 on Client 8 took 3.75s  
- Training Round 1 on Client 9 took 2.63s  
- Training Round 1 on Client 10 took 3.37s  


Agent 1
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.45 | ransomware\_file\_extension\_hide |
| bdvl               |       0.44 | rootkit\_sanitizer              |
| beurk              |       0    | rootkit\_sanitizer              |
| the\_tick           |     100    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.61 | ransomware\_file\_extension\_hide |
| bdvl               |       0    | rootkit\_sanitizer              |
| beurk              |       0    | rootkit\_sanitizer              |
| the\_tick           |     100    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 3
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |       99.3 | ransomware\_file\_extension\_hide |
| bdvl               |        0   | rootkit\_sanitizer              |
| beurk              |        0   | rootkit\_sanitizer              |
| the\_tick           |      100   | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      100   | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      100   | cnc\_ip\_shuffle                 |
| data\_leak\_2        |      100   | cnc\_ip\_shuffle                 |  


Agent 4
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      97.86 | ransomware\_file\_extension\_hide |
| bdvl               |       0    | rootkit\_sanitizer              |
| beurk              |       0    | rootkit\_sanitizer              |
| the\_tick           |     100    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 5
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.93 | ransomware\_file\_extension\_hide |
| bdvl               |       0    | rootkit\_sanitizer              |
| beurk              |       0    | rootkit\_sanitizer              |
| the\_tick           |     100    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 6
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      99.36 | ransomware\_file\_extension\_hide |
| bdvl               |       0    | rootkit\_sanitizer              |
| beurk              |       0    | rootkit\_sanitizer              |
| the\_tick           |     100    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 7
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |       99.3 | ransomware\_file\_extension\_hide |
| bdvl               |        0   | rootkit\_sanitizer              |
| beurk              |        0   | rootkit\_sanitizer              |
| the\_tick           |      100   | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      100   | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      100   | cnc\_ip\_shuffle                 |
| data\_leak\_2        |      100   | cnc\_ip\_shuffle                 |  


Agent 8
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.61 | ransomware\_file\_extension\_hide |
| bdvl               |       0    | rootkit\_sanitizer              |
| beurk              |       0    | rootkit\_sanitizer              |
| the\_tick           |     100    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 9
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      99.25 | ransomware\_file\_extension\_hide |
| bdvl               |       0.09 | rootkit\_sanitizer              |
| beurk              |       0    | rootkit\_sanitizer              |
| the\_tick           |      99.93 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.73 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |      99.88 | cnc\_ip\_shuffle                 |  


Agent 10
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |       99.3 | ransomware\_file\_extension\_hide |
| bdvl               |        0   | rootkit\_sanitizer              |
| beurk              |        0   | rootkit\_sanitizer              |
| the\_tick           |      100   | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      100   | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      100   | cnc\_ip\_shuffle                 |
| data\_leak\_2        |      100   | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      99.62 | ransomware\_file\_extension\_hide |
| bdvl               |       0    | rootkit\_sanitizer              |
| beurk              |       0    | rootkit\_sanitizer              |
| the\_tick           |     100    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 2/10  
  
- Training Round 1 on Client 1 took 3.11s  
- Training Round 1 on Client 2 took 2.84s  
- Training Round 1 on Client 3 took 2.57s  
- Training Round 1 on Client 4 took 2.4s  
- Training Round 1 on Client 5 took 2.69s  
- Training Round 1 on Client 6 took 3.07s  
- Training Round 1 on Client 7 took 2.44s  
- Training Round 1 on Client 8 took 2.99s  
- Training Round 1 on Client 9 took 3.3s  
- Training Round 1 on Client 10 took 3.34s  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.82 | ransomware\_file\_extension\_hide |
| bdvl               |      63.51 | rootkit\_sanitizer              |
| beurk              |      94.8  | rootkit\_sanitizer              |
| the\_tick           |       4.11 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |       2.58 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      12.33 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |      20.68 | cnc\_ip\_shuffle                 |  
- Training Round 1 on Client 1 took 3.26s  
- Training Round 1 on Client 2 took 3.07s  
- Training Round 1 on Client 3 took 2.68s  
- Training Round 1 on Client 4 took 2.44s  
- Training Round 1 on Client 5 took 2.89s  
- Training Round 1 on Client 6 took 3.36s  
- Training Round 1 on Client 7 took 2.67s  
- Training Round 1 on Client 8 took 3.21s  
- Training Round 1 on Client 9 took 2.69s  
- Training Round 1 on Client 10 took 2.68s  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      99.3  | ransomware\_file\_extension\_hide |
| bdvl               |       0    | rootkit\_sanitizer              |
| beurk              |       0    | rootkit\_sanitizer              |
| the\_tick           |     100    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |      97.32 | cnc\_ip\_shuffle                 |  
- Training Round 2 on Client 1 took 2.73s  
- Training Round 2 on Client 2 took 2.67s  
- Training Round 2 on Client 3 took 3.21s  
- Training Round 2 on Client 4 took 2.96s  
- Training Round 2 on Client 5 took 2.99s  
- Training Round 2 on Client 6 took 2.69s  
- Training Round 2 on Client 7 took 3.42s  
- Training Round 2 on Client 8 took 3.75s  
- Training Round 2 on Client 9 took 3.43s  
- Training Round 2 on Client 10 took 3.35s  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      99.25 | ransomware\_file\_extension\_hide |
| bdvl               |       0    | rootkit\_sanitizer              |
| beurk              |       0    | rootkit\_sanitizer              |
| the\_tick           |     100    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
- Training Round 3 on Client 1 took 2.52s  
- Training Round 3 on Client 2 took 2.49s  
- Training Round 3 on Client 3 took 2.67s  
- Training Round 3 on Client 4 took 2.56s  
- Training Round 3 on Client 5 took 2.46s  
- Training Round 3 on Client 6 took 3.09s  
- Training Round 3 on Client 7 took 2.64s  
- Training Round 3 on Client 8 took 3.23s  
- Training Round 3 on Client 9 took 2.62s  
- Training Round 3 on Client 10 took 3.12s  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      99.09 | ransomware\_file\_extension\_hide |
| bdvl               |      13.2  | rootkit\_sanitizer              |
| beurk              |       0    | rootkit\_sanitizer              |
| the\_tick           |     100    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
- Training Round 4 on Client 1 took 2.63s  
- Training Round 4 on Client 2 took 2.93s  
- Training Round 4 on Client 3 took 2.77s  
- Training Round 4 on Client 4 took 2.59s  
- Training Round 4 on Client 5 took 3.68s  
- Training Round 4 on Client 6 took 2.95s  
- Training Round 4 on Client 7 took 3.12s  
- Training Round 4 on Client 8 took 2.97s  
- Training Round 4 on Client 9 took 2.76s  
- Training Round 4 on Client 10 took 2.55s  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.71 | ransomware\_file\_extension\_hide |
| bdvl               |      60.05 | rootkit\_sanitizer              |
| beurk              |       0    | rootkit\_sanitizer              |
| the\_tick           |     100    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
- Training Round 5 on Client 1 took 2.93s  
- Training Round 5 on Client 2 took 2.71s  
- Training Round 5 on Client 3 took 3.21s  
- Training Round 5 on Client 4 took 2.88s  
- Training Round 5 on Client 5 took 2.6s  
- Training Round 5 on Client 6 took 2.53s  
- Training Round 5 on Client 7 took 2.44s  
- Training Round 5 on Client 8 took 3.25s  
- Training Round 5 on Client 9 took 2.75s  
- Training Round 5 on Client 10 took 2.74s  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.61 | ransomware\_file\_extension\_hide |
| bdvl               |      64.66 | rootkit\_sanitizer              |
| beurk              |       0    | rootkit\_sanitizer              |
| the\_tick           |     100    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
- Training Round 6 on Client 1 took 2.59s  
- Training Round 6 on Client 2 took 3.01s  
- Training Round 6 on Client 3 took 3.24s  
- Training Round 6 on Client 4 took 2.48s  
- Training Round 6 on Client 5 took 2.59s  
- Training Round 6 on Client 6 took 3.49s  
- Training Round 6 on Client 7 took 3.01s  
- Training Round 6 on Client 8 took 2.61s  
- Training Round 6 on Client 9 took 2.49s  
- Training Round 6 on Client 10 took 3.47s  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.45 | ransomware\_file\_extension\_hide |
| bdvl               |      74.31 | rootkit\_sanitizer              |
| beurk              |       0    | rootkit\_sanitizer              |
| the\_tick           |     100    | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
- Training Round 7 on Client 1 took 2.9s  
- Training Round 7 on Client 2 took 3.1s  
- Training Round 7 on Client 3 took 2.69s  
- Training Round 7 on Client 4 took 2.87s  
- Training Round 7 on Client 5 took 2.74s  
- Training Round 7 on Client 6 took 2.51s  
- Training Round 7 on Client 7 took 2.23s  
- Training Round 7 on Client 8 took 2.41s  
- Training Round 7 on Client 9 took 2.84s  
- Training Round 7 on Client 10 took 2.79s  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.55 | ransomware\_file\_extension\_hide |
| bdvl               |      92.74 | rootkit\_sanitizer              |
| beurk              |       0.21 | rootkit\_sanitizer              |
| the\_tick           |      99.87 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
- Training Round 8 on Client 1 took 2.84s  
- Training Round 8 on Client 2 took 2.38s  
- Training Round 8 on Client 3 took 2.62s  
- Training Round 8 on Client 4 took 2.54s  
- Training Round 8 on Client 5 took 3.05s  
- Training Round 8 on Client 6 took 2.37s  
- Training Round 8 on Client 7 took 2.84s  
- Training Round 8 on Client 8 took 2.51s  
- Training Round 8 on Client 9 took 2.76s  
- Training Round 8 on Client 10 took 2.99s  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.66 | ransomware\_file\_extension\_hide |
| bdvl               |      98.32 | rootkit\_sanitizer              |
| beurk              |       1.44 | rootkit\_sanitizer              |
| the\_tick           |      99.09 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      99.18 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
- Training Round 9 on Client 1 took 2.73s  
- Training Round 9 on Client 2 took 2.98s  
- Training Round 9 on Client 3 took 2.49s  
- Training Round 9 on Client 4 took 2.37s  
- Training Round 9 on Client 5 took 2.83s  
- Training Round 9 on Client 6 took 2.71s  
- Training Round 9 on Client 7 took 2.71s  
- Training Round 9 on Client 8 took 2.47s  
- Training Round 9 on Client 9 took 2.5s  
- Training Round 9 on Client 10 took 2.25s  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.66 | ransomware\_file\_extension\_hide |
| bdvl               |      99.11 | rootkit\_sanitizer              |
| beurk              |       4.86 | rootkit\_sanitizer              |
| the\_tick           |      97.52 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      97.3  | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
- Training Round 10 on Client 1 took 2.36s  
- Training Round 10 on Client 2 took 2.53s  
- Training Round 10 on Client 3 took 2.53s  
- Training Round 10 on Client 4 took 2.11s  
- Training Round 10 on Client 5 took 3.21s  
- Training Round 10 on Client 6 took 2.14s  
- Training Round 10 on Client 7 took 2.36s  
- Training Round 10 on Client 8 took 2.52s  
- Training Round 10 on Client 9 took 2.46s  
- Training Round 10 on Client 10 took 2.38s  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.71 | ransomware\_file\_extension\_hide |
| bdvl               |      99.11 | rootkit\_sanitizer              |
| beurk              |       6.02 | rootkit\_sanitizer              |
| the\_tick           |      96.54 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      95.89 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.91 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
![graph](round-10\_agent-01\_learning-curve.png)  
![graph](round-10\_agent-02\_learning-curve.png)  
![graph](round-10\_agent-03\_learning-curve.png)  
![graph](round-10\_agent-04\_learning-curve.png)  
![graph](round-10\_agent-05\_learning-curve.png)  
![graph](round-10\_agent-06\_learning-curve.png)  
![graph](round-10\_agent-07\_learning-curve.png)  
![graph](round-10\_agent-08\_learning-curve.png)  
![graph](round-10\_agent-09\_learning-curve.png)  
![graph](round-10\_agent-10\_learning-curve.png)  
