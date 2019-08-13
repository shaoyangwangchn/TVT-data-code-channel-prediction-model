% clc;
% clear;
% train_loss = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_10.xlsx', 'A:A');
% validation_loss = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_10.xlsx', 'B:B');
% test_loss = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_10.xlsx', 'C:C');
% train_prediction = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_10.xlsx', 'D:D');
% validation_prediction = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_10.xlsx', 'E:E');
% total_prediction = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_10.xlsx', 'F:F');
% test_prediction = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_10.xlsx', 'G:G');
% train_prediction_pre = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_10.xlsx', 'H:H');
% validation_prediction_pre = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_10.xlsx', 'I:I');
% real_total = load('channel_data_90_10.mat');
% real_total = real_total.h;
% real_total = real_total(1,1:8000);
% real_total = abs(real_total);
% 
% train_loss_16 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_16.xlsx', 'A:A');
% validation_loss_16 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_16.xlsx', 'B:B');
% test_loss_16 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_16.xlsx', 'C:C');
% train_prediction_16 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_16.xlsx', 'D:D');
% validation_prediction_16 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_16.xlsx', 'E:E');
% total_prediction_16 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_16.xlsx', 'F:F');
% test_prediction_16 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_16.xlsx', 'G:G');
% train_prediction_pre_16 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_16.xlsx', 'H:H');
% validation_prediction_pre_16 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_16.xlsx', 'I:I');
% 
% train_loss_20 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_20.xlsx', 'A:A');
% validation_loss_20 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_20.xlsx', 'B:B');
% test_loss_20 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_10.xlsx', 'C:C');
% train_prediction_20 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_20.xlsx', 'D:D');
% validation_prediction_20 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_20.xlsx', 'E:E');
% total_prediction_20 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_20.xlsx', 'F:F');
% test_prediction_20 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_20.xlsx', 'G:G');
% train_prediction_pre_20 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_20.xlsx', 'H:H');
% validation_prediction_pre_20 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_20.xlsx', 'I:I');
% 
% train_begin = 1;
% train_end = 4800;
% validation_end = 6400;
% train_num_pre = 200;

%% 总预测图
figure(1)
subplot(321)
t_real_total = 1:8000;
plot(t_real_total, real_total,'LineWidth', 1.25);
hold on
t_train_prediction_pre = 6:train_end;
plot(t_train_prediction_pre, train_prediction_pre, 'LineWidth', 1.25);
hold on
t_validation_prediction_pre = train_end+1:validation_end;
plot(t_validation_prediction_pre, validation_prediction_pre, 'LineWidth', 1.25);
hold on
t_test_prediction = validation_end+1:8000;
plot(t_test_prediction, test_prediction, 'LineWidth', 1.25)
hold on
t_prediction_total = 6:8000;
plot(t_prediction_total, total_prediction, 'LineWidth', 1.25);
hold on
legend('Real channel data', 'Prediction based on train data by pre-training model', 'Prediction based on validation data by pre-training model', 'Online prediction by IL', 'Prediction based on total data by the IL model when time-slot is 8000');
plot([4800, 4800], [0, 1],'r', 'LineWidth', 1.25 )
plot([6400, 6400], [0, 1],'r', 'LineWidth', 1.25 )
grid on
ylabel('|h_{1}(t)|, l=10')

%% 总误差图
subplot(322)
% error_train_pre = zeros(1, train_end-5);
% error_validation_pre = zeros(1:validation_end-train_end);
t_error_train_pre = 6:train_end;
t_error_validation_pre = train_end+1:validation_end;
t_error_test_IL = validation_end+1:8000;
t_error_total = 6:8000;
error_train_pre = train_prediction_pre - real_total(6: train_end)';
error_validation_pre = validation_prediction_pre - real_total(train_end+1:validation_end)';
error_test_IL = test_prediction - real_total(6401:8000)';
error_total = total_prediction - real_total(6:8000)';
scatter(t_error_train_pre, error_train_pre, '.');
hold on 
scatter(t_error_validation_pre, error_validation_pre, '.');
hold on 
scatter(t_error_test_IL, error_test_IL, '.');
hold on 
scatter(t_error_total, error_total, '.');
hold on 
plot([1, 8000],[0, 0],'c-', 'LineWidth', 1.75)
hold on
legend('Error based on train data', 'Error based on validation data','Online prediction error', 'Error based total data by by the IL model when time-slot is 8000', 'Zero-error reference');
grid on
ylabel('Error')

%% 总预测图
subplot(323)
t_real_total = 1:8000;
plot(t_real_total, real_total,'LineWidth', 1.25);
hold on
t_train_prediction_pre = 6:train_end;
plot(t_train_prediction_pre, train_prediction_pre_16, 'LineWidth', 1.25);
hold on
t_validation_prediction_pre = train_end+1:validation_end;
plot(t_validation_prediction_pre, validation_prediction_pre_16, 'LineWidth', 1.25);
hold on
t_test_prediction = validation_end+1:8000;
plot(t_test_prediction, test_prediction_16, 'LineWidth', 1.25)
hold on
t_prediction_total = 6:8000;
plot(t_prediction_total, total_prediction_16, 'LineWidth', 1.25);
hold on
legend('Real channel data', 'Prediction based on train data by pre-training model', 'Prediction based on validation data by pre-training model', 'Online prediction by IL', 'Prediction based on total data by the IL model when time-slot is 8000');
plot([4800, 4800], [-0.2, 1],'r', 'LineWidth', 1.25 )
plot([6400, 6400], [-0.2, 1],'r', 'LineWidth', 1.25 )
grid on
ylabel('|h_{1}(t)|, l=16')

%% 总误差图
subplot(324)
% error_train_pre = zeros(1, train_end-5);
% error_validation_pre = zeros(1:validation_end-train_end);
t_error_train_pre = 6:train_end;
t_error_validation_pre = train_end+1:validation_end;
t_error_test_IL = validation_end+1:8000;
t_error_total = 6:8000;
error_train_pre_16 = train_prediction_pre_16 - real_total(6: train_end)';
error_validation_pre_16 = validation_prediction_pre_16 - real_total(train_end+1:validation_end)';
error_test_IL_16 = test_prediction_16 - real_total(6401:8000)';
error_total_16 = total_prediction_16 - real_total(6:8000)';
scatter(t_error_train_pre, error_train_pre_16, '.');
hold on 
scatter(t_error_validation_pre, error_validation_pre_16, '.');
hold on 
scatter(t_error_test_IL, error_test_IL_16, '.');
hold on 
scatter(t_error_total, error_total_16, '.');
hold on 
plot([1, 8000],[0, 0],'c-', 'LineWidth', 1.75)
hold on
legend('Error based on train data', 'Error based on validation data','Online prediction error', 'Error based total data by by the IL model when time-slot is 8000', 'Zero-error reference');
grid on
ylabel('Error')

%% 总预测图
subplot(325)
t_real_total_20 = 1:8000;
plot(t_real_total, real_total,'LineWidth', 1.25);
hold on
t_train_prediction_pre = 6:train_end;
plot(t_train_prediction_pre, train_prediction_pre_20, 'LineWidth', 1.25);
hold on
t_validation_prediction_pre = train_end+1:validation_end;
plot(t_validation_prediction_pre, validation_prediction_pre_20, 'LineWidth', 1.25);
hold on
t_test_prediction = validation_end+1:8000;
plot(t_test_prediction, test_prediction_20, 'LineWidth', 1.25)
hold on
t_prediction_total = 6:8000;
plot(t_prediction_total, total_prediction_20, 'LineWidth', 1.25);
hold on
legend('Real channel data', 'Prediction based on train data by pre-training model', 'Prediction based on validation data by pre-training model', 'Online prediction by IL', 'Prediction based on total data by the IL model when time-slot is 8000');
plot([4800, 4800], [-0.2, 1],'r', 'LineWidth', 1.25 )
plot([6400, 6400], [-0.2, 1],'r', 'LineWidth', 1.25 )
grid on
xlabel('Time-slot, v=90 km/h')
ylabel('|h_{1}(t)|, l=20')

%% 总误差图
subplot(326)
% error_train_pre = zeros(1, train_end-5);
% error_validation_pre = zeros(1:validation_end-train_end);
t_error_train_pre = 6:train_end;
t_error_validation_pre = train_end+1:validation_end;
t_error_test_IL = validation_end+1:8000;
t_error_total = 6:8000;
error_train_pre_20 = train_prediction_pre_20 - real_total(6: train_end)';
error_validation_pre_20 = validation_prediction_pre_20 - real_total(train_end+1:validation_end)';
error_test_IL_20 = test_prediction_20 - real_total(6401:8000)';
error_total_20 = total_prediction_20 - real_total(6:8000)';
scatter(t_error_train_pre, error_train_pre_20, '.');
hold on 
scatter(t_error_validation_pre, error_validation_pre_20, '.');
hold on 
scatter(t_error_test_IL, error_test_IL_20, '.');
hold on 
scatter(t_error_total, error_total_20, '.');
hold on 
plot([1, 8000],[0, 0],'c-', 'LineWidth', 1.75)
hold on
legend('Error based on train data', 'Error based on validation data','Online prediction error', 'Error based total data by by the IL model when time-slot is 8000', 'Zero-error reference');
grid on
xlabel('Time-slot')
ylabel('Error')


% %% 总loss图
% train_loss_pre = train_loss;
% figure(10)
% t_train_loss_pre = 1:359;
% semilogy(t_train_loss_pre, train_loss_pre, 'LineWidth', 1.25);
% hold on
% t_valadation_loss = 1:200;
% semilogy(t_valadation_loss, validation_loss, 'LineWidth', 1.25);
% hold on
% legend('Train loss', 'Validation loss');
% plot([200, 200], [10^(-10), 1],'r', 'LineWidth', 1.25 )
% grid on
% xlabel('Pre-training and IL step')
% ylabel('Loss')

