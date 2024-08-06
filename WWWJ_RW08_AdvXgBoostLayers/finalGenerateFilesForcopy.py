import sys, os


# Algs=['finalAttentionsAdvMemTx','finalAttentionsWht','finalAttentionsQVK','finalAttentionsWhtMemTx','finalAttentionsQVKMemTx',
#       'finalAttentionsAdv','finalLSTM']
Algs=['xgBoost_0Layer','xgBoost_1Layer','xgBoost_2Layer']
Other_Stras=['']


abbv='Class'
TestGroups=['34']
testgroup='31'

totalfiles=1

classList=[2,4,6,8]


def GeneratePythonfiles(Testgroups,algname,Other_Stras):
    sourcefile='2ClassSet31'+algname+Other_Stras[0]+'.py'
    for testgroup in Testgroups:
        for stra in Other_Stras:
            for filecount in range(totalfiles):
                for classLabel in classList:
                    f = open(sourcefile, 'r')
                    flist = f.readlines()
                    f.close()
                    for i in range(len(flist)):
                        if 'clusterNumber=2' in flist[i]:
                            flist[i] = flist[i].replace('2', str(classLabel))


                        if 'TestGroup=' in flist[i]:
                            flist[i] = flist[i].replace('31', testgroup)
                        if 'addaccount' in flist[i]:
                            flist[i] = flist[i].replace('0', str(filecount))
                    filename = str(classLabel)+abbv+'Set'+testgroup+algname+stra+str(filecount+1)+'.py'
                    f = open(filename, 'w+')
                    f.writelines(flist)
                    f.close()

def GenerateShfiles(Testgroups,algname,Other_Stras):
    sourcefile='2ClassSet31'+algname+Other_Stras[0]+'.sh'
    for testgroup in Testgroups:
        for stra in Other_Stras:   


            
            for filecount in range(totalfiles):
                for classLabel in classList:
                    f = open(sourcefile, 'r')
                    flist = f.readlines()
                    f.close()
                    for i in range(len(flist)):
                        if algname in flist[i]:
                            flist[i] = flist[i].replace(algname,algname+str(filecount+1))
                        if 'Set31' in flist[i]:
                            flist[i] = flist[i].replace('31',testgroup)

                        if 'Class' in flist[i]:
                            flist[i] = flist[i].replace('2Class', str(classLabel) + 'Class')
                    filename =str(classLabel)+abbv+ 'Set'+testgroup+algname+stra+str(filecount+1)+'.sh'
                    f = open(filename, 'w+')
                    f.writelines(flist)
                    f.close()

for alg in Algs:
    GeneratePythonfiles(TestGroups,alg,Other_Stras)
    GenerateShfiles(TestGroups, alg,Other_Stras)



fileList=os.listdir()


AAAContents = []
AAAContents.append('#!/bin/bash\n')
AAAContents.append('# SBATCH --time=1:20:00\n')
for filename in fileList:
    if filename.endswith('.sh') and 'AAA' not in filename:
        AAAContents.append('sbatch '+filename+'\n')
f = open('AAA.sh', 'w+')
f.writelines(AAAContents)
