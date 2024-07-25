# BilliardEye
A sport video analysis software for billiard matches.

To develop the system, the following tasks were solved:
- Table segmentation
- Table sides recognition
- Balls localization and segmentatation
- Balls identification
- Balls tracking
- Map rendering
- System performance evaluation

Full details on the project specifics and development, system outputs and program execution can be found in the [report](Report.pdf).

Example of system output:

https://github.com/user-attachments/assets/8b2f483b-b78b-45da-a166-a4064f8a1e71

---
## Execution instructions
Project building:
```
git clone https://github.com/mattia-toffolon/BilliardEye.git
cd BilliardEye
mkdir build
cd build
cmake ..
make
```
System setup:
```
cd ..
unzip samples.zip
```
Program execution:
```
cd build
./outMain "path-to-video" "path-to-first-frame" "path-to-output-dir"
./perfMain "path-to-samples-folder" "number-of-samples"
```
