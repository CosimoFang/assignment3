CS5300-Project3-RealTime2DObjectRecognition
Minghao Liu

os win11
ide vs studio

Khoury wiki:
https://wiki.khoury.northeastern.edu/x/DYUAC
Youtube Link:
https://www.youtube.com/watch?v=2TCIKWptWWI

filter.h/filter.cpp include filtering, thresholding, morphological operation, clustering , cal the distance.
vidDisplay.cpp: the mainfunction of the program, read img from camera, read txt in the directory, and out put
db.text: contains the current feature and database file
how to use:
1 use cd to the dir and use cmd to run the exe file.
cmd : project1.exe

2 press l to load new feature with cmd input name into the database

extension:
3 if the new img is unknown for 300 frames, it will detect and ask user to manually input a new name for it
4 detect up t0 16 different objects in the db.txt
5 implement two additonal distance matrix Minkowski and Manatan
6 better gui in cmd
