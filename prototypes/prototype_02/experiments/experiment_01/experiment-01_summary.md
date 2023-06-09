  
# Prototype 1 (Experiment 1)  
---  
  
Executed on 13.03.2023  
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
- 2879 samples of Behavior.NORMAL  
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
- 2879 samples of Behavior.NORMAL  
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
  
- Training Round 1 on Client 1 took 57.39s  
- Training Round 1 on Client 2 took 65.85s  
![graph](round-01\_agent-01\_learning-curve.png)  
![graph](round-01\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.02 | ransomware\_file\_extension\_hide |
| bdvl               |      98.49 | rootkit\_sanitizer              |
| beurk              |      18.69 | rootkit\_sanitizer              |
| the\_tick           |      81.65 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      81.34 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      96.84 | ransomware\_file\_extension\_hide |
| bdvl               |      98.49 | rootkit\_sanitizer              |
| beurk              |      32.44 | rootkit\_sanitizer              |
| the\_tick           |      65.77 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      70.31 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      95.77 | ransomware\_file\_extension\_hide |
| bdvl               |      98.58 | rootkit\_sanitizer              |
| beurk              |      27.86 | rootkit\_sanitizer              |
| the\_tick           |      70.54 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      74.77 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 2/10  
  
- Training Round 2 on Client 1 took 52.39s  
- Training Round 2 on Client 2 took 69.34s  
![graph](round-02\_agent-01\_learning-curve.png)  
![graph](round-02\_agent-02\_learning-curve.png)  


Agent 1
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      97.96 | ransomware\_file\_extension\_hide |
| bdvl               |      98.32 | rootkit\_sanitizer              |
| beurk              |      72.21 | rootkit\_sanitizer              |
| the\_tick           |      33.57 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      29.58 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.82 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Agent 2
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.5  | ransomware\_file\_extension\_hide |
| bdvl               |      98.58 | rootkit\_sanitizer              |
| beurk              |      36.76 | rootkit\_sanitizer              |
| the\_tick           |      61.72 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      63.97 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |      99.91 | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  


Global Agent
  
| Behavior           |   Accuracy | Objective                      |
|:-------------------|-----------:|:-------------------------------|
| ransomware\_poc     |      98.39 | ransomware\_file\_extension\_hide |
| bdvl               |      98.49 | rootkit\_sanitizer              |
| beurk              |      48.6  | rootkit\_sanitizer              |
| the\_tick           |      49.25 | cnc\_ip\_shuffle                 |
| backdoor\_jakoritar |      48.24 | cnc\_ip\_shuffle                 |
| data\_leak\_1        |     100    | cnc\_ip\_shuffle                 |
| data\_leak\_2        |     100    | cnc\_ip\_shuffle                 |  
  
<div style="page-break-after: always;"></div>  
  
---  
### Training Round 3/10  
  
