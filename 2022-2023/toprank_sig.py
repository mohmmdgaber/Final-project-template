import os
import tensorflow.compat.v1 as tff
tff.compat.v1.disable_eager_execution()
import numpy as np
import joblib
from SignatureDataGenerator import *

FLAGS = None

def deepnn(x, train):
  with tff.name_scope('reshape'):
    x_image = tff.reshape(x, [-1, 2048])
  with tff.name_scope('fc1'):
    W_fc1 = weight_variable([2048, 1024])
    b_fc1 = bias_variable([1024]) 
    h_fc1 = tff.nn.relu(tff.matmul(x_image, W_fc1) + b_fc1)
    
  with tff.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 512])
    b_fc2 = bias_variable([512]) 
    h_fc2 = tff.nn.relu(tff.matmul(h_fc1, W_fc2) + b_fc2)

  with tff.name_scope('fc3'):
    W_fc3 = weight_variable([512, 128])
    b_fc3 = bias_variable([128]) 
    h_fc3 = tff.nn.relu(tff.matmul(h_fc2, W_fc3) + b_fc3)    
    
  with tff.name_scope('fc4'):
    W_fc4 = weight_variable([128, 1])
    b_fc4 = bias_variable([1])
    y_conv = tff.sigmoid(tff.matmul(h_fc3, W_fc4) + b_fc4)
    return y_conv, h_fc3

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tff.truncated_normal(shape, stddev=0.05)
  return tff.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tff.constant(0.1, shape=shape)
  return tff.Variable(initial)

def get_batch(name_pos, label_pos, name_neg, label_neg, batch_size_p, batch_size_n, now_batch_p, now_batch_n, total_batch_n):
    image_batch_pos = name_pos[now_batch_p*batch_size_p:(now_batch_p+1)*batch_size_p,:]
    label_batch_pos = label_pos[now_batch_p*batch_size_p:(now_batch_p+1)*batch_size_p]
    image_batch_neg = name_neg[now_batch_n*batch_size_n:(now_batch_n+1)*batch_size_n,:]
    label_batch_neg = label_neg[now_batch_n*batch_size_n:(now_batch_n+1)*batch_size_n]        
    batchdata=np.concatenate((image_batch_pos,image_batch_neg),axis=0)        
    batchlabel=np.concatenate((label_batch_pos,label_batch_neg),axis=0)
    return batchdata, batchlabel

def loss_with_top_rank(logits, labels, p):
    
  with tff.name_scope('toprank'):
      
      loss = tff.zeros([1,1])
            
      num = tff.constant(1/5)
            
      index1 = tff.where(tff.equal(labels,1))
            
      indices1 = index1[:,0]
            
      index2 = tff.where(tff.not_equal(labels,1))
            
      indices2 = index2[:,0] 
            
      cs=tff.constant(1.0)
        
      p1 = p
        
      p2 = cs/p1
            
      sum_n = tff.zeros([1,1])
            
      sum_p = tff.zeros([1,1])
            
      l = tff.zeros([1,1])
            
      feature = tff.zeros([1,1])
            
      norm_feature = tff.zeros([1,1])
            
      l_p = tff.zeros([1,1])
            
      for u in range(5):
            
          for v in range(40):
                  
              feature = tff.subtract(logits[indices1[u]],logits[indices2[v]])
                
              norm_feature =tff.log(1+tff.exp(-feature))
                
              l_p = tff.pow(norm_feature,p1)
                
              sum_n = sum_n+l_p
    
          l = tff.pow(sum_n,p2)
              
          sum_p = sum_p+l
              
      loss = tff.multiply(sum_p,num)         
          
      return loss
  
def main(_):   
    for fold in range(5):
        p_array=[2,4,8,16,32]
        num_p=p_array[fold]
        p_value=np.float32(num_p)
        p_value=np.reshape(p_value,[1])
        tff.reset_default_graph()
        model_name = 'hindi_model'+'_p_' + str(num_p)
        training = tff.placeholder(tff.bool)    
        x = tff.placeholder(tff.float32, [None, 2048])
        y_ = tff.placeholder(tff.int64, [None,1])
        y_conv,fc1 = deepnn(x, training)

        with tff.name_scope('loss'):
    
            toprankloss = loss_with_top_rank(logits=y_conv, labels=y_, p=p_value)
    
            toprankloss = tff.reshape(toprankloss,[])

            initial_loss= tff.constant(1e-14)

            toprankloss = tff.add(toprankloss,initial_loss)

            loss_summary = tff.summary.scalar('loss', toprankloss)      

        global_step = tff.Variable(0, trainable=False)
        initial_learning_rate = 0.0001
        learning_rate = tff.train.exponential_decay(initial_learning_rate,
                                           global_step,
                                           decay_steps=100,decay_rate=0.90)
        with tff.name_scope('adam_optimizer'):
            update_ops = tff.get_collection(tff.GraphKeys.UPDATE_OPS)
        with tff.control_dependencies(update_ops):
            train_step = tff.train.AdamOptimizer(learning_rate).minimize(toprankloss)
        add_global = global_step.assign_add(1)

        sess=tff.Session()
        train_writer = tff.summary.FileWriter('/home/xiaotong/Desktop/Experiments/Codes/Reliable_methods/TopRankNN_related/summary/p_' + str(num_p)+ '/'+ 'p_' + str(num_p)+'_fc_hindi_3', sess.graph)
        sess.run(tff.global_variables_initializer())
        saver = tff.train.Saver()                  
        step=0
        for epoch in range(10):
            file=joblib.load('/home/xiaotong/Desktop/Experiments/Codes/Signature/dataset/BHSig260/'+di.dataset+'/resized/')
            train=file['train_pairs']
            traindata_pos,traindata_neg=train['G-G'],train['G-F']
            traindata_pos=np.array(traindata_pos)
            traindata_neg=np.array(traindata_neg)
            train_g_len=len(traindata_pos)
            train_f_len=len(traindata_neg)
            trainlabel_p=np.ones([train_g_len])
            trainlabel_n=np.zeros([train_f_len])
            now_batch_p=0
            now_batch_n=0
            batch_size_p=5
            batch_size_n=40
            total_batch_n=np.int(train_f_len/batch_size_n)+1
            num_of_pos=np.int(train_g_len/batch_size_p)
            num_of_neg=np.int(train_f_len/batch_size_n)
            train_p=traindata_pos
            train_n=traindata_neg
            for i in range(num_of_pos):
                now_batch_p=i
                if i < num_of_neg:
                    x_batch_train,y_batch_train=get_batch(train_p,trainlabel_p,train_n,trainlabel_n,batch_size_p,batch_size_n,now_batch_p,now_batch_n,total_batch_n)
                if i >= num_of_neg:
                    if now_batch_n % num_of_neg == 0:
                        now_batch_n = 0
                        train_n = traindata_neg
                    x_batch_train,y_batch_train = get_batch(train_p,trainlabel_p,train_n,trainlabel_n,batch_size_p,batch_size_n,now_batch_p,now_batch_n,total_batch_n)
                x_batch_train=np.float32(x_batch_train)
                y_batch_train=np.reshape(y_batch_train,[45,1])
                g, rate = sess.run([add_global, learning_rate])
                loss,losssummary, _ = sess.run([toprankloss,loss_summary,train_step], feed_dict={x: x_batch_train, y_ : y_batch_train, training:True})
                train_writer.add_summary(losssummary, step)
                print("step: {}, train loss: {:g}".format(step, loss))
                now_batch_n=now_batch_n+1                
                step=step+1
            if (epoch % 1 == 0):
                print("Saving checkpoint")
                saver.save(sess, '/home/xiaotong/Desktop/Experiments/Codes/Reliable_methods/TopRankNN_related/models/p_'+str(num_p)+'/' + model_name + '_fc.ckpt')
            
            
if __name__ == '__main__':
    tff.app.run()


