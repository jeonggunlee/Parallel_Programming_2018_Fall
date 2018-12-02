#read string data

cudaResult<-readLines("cuda.txt",encoding = "UTF-8")
tensorResult<-readLines("tensor.txt",encoding = "UTF-8")

# splite string data
cudaResult<-strsplit(cudaResult,split = " ")
tensorResult<-strsplit(tensorResult,split = " ")

# change format
cuda<-unlist(cudaResult)
tensor<-unlist(tensorResult)

# change string to numeric(dpuble)
cuda<-as.numeric(cuda)
tensor<-as.numeric(tensor)

# datas mean
mean(cuda)
mean(tensor)

# var
var(cuda)
var(tensor)

# standard
sd(cuda)
sd(tensor)

# test
var.test(cuda,tensor) # p-value 0.9975
t.test(cuda,tensor,paired = TRUE) # p-value 2.2e-16
# so cuda and tensor result are different

head(tensor)
head(cuda)

differ<-cuda-tensor

mean(differ)
sd(differ)
max(abs(differ))
