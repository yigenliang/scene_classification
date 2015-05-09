import os,sys
sys.path.append(os.path.dirname(__file__)+"/../assets/libsvm-3.20/python")
from svmutil import *
def svm_sort(train_data_path,test_data_path,output_path):
    output_data=open(output_path,"w")
    train_y,train_x=svm_read_problem(train_data_path)
    model=svm_train(train_y,train_x,'-c 1 -h 1')

    test_y,test_x=svm_read_problem(test_data_path)
    label,acc,val=svm_predict(test_y,test_x,model)

    for label in label:
        output_data.writelines(label+"\n")
    output_data.close()
if __name__=='__main__':
    svm_sort("train","test","output")