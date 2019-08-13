% clc;
% clear;
% train_loss = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\single-point prediction online\prediction_180_4800_1600_1600.xlsx', 'A:A');
% validation_loss = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\single-point prediction online\prediction_180_4800_1600_1600.xlsx', 'B:B');
% test_loss = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\single-point prediction online\prediction_180_4800_1600_1600.xlsx', 'C:C');
% train_prediction = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\single-point prediction online\prediction_180_4800_1600_1600.xlsx', 'D:D');
% validation_prediction = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\single-point prediction online\prediction_180_4800_1600_1600.xlsx', 'E:E');
% total_prediction = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\single-point prediction online\prediction_180_4800_1600_1600.xlsx', 'F:F');
% test_prediction = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\single-point prediction online\prediction_180_4800_1600_1600.xlsx', 'G:G');
% train_prediction_pre = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\single-point prediction online\prediction_180_4800_1600_1600.xlsx', 'H:H');
% validation_prediction_pre = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\single-point prediction online\prediction_180_4800_1600_1600.xlsx', 'I:I');
% real_total = load('channel_data10.mat');
% real_total = real_total.h;
% real_total = real_total(1,1:8000);
% real_total = abs(real_total);
% 
% 
% train_loss_360 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\single-point prediction online\prediction_360_4800_1600_1600.xlsx', 'A:A');
% validation_loss_360 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\single-point prediction online\prediction_360_4800_1600_1600.xlsx', 'B:B');
% test_loss_360 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\single-point prediction online\prediction_360_4800_1600_1600.xlsx', 'C:C');
% train_prediction_360 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\single-point prediction online\prediction_360_4800_1600_1600.xlsx', 'D:D');
% validation_prediction_360 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\single-point prediction online\prediction_360_4800_1600_1600.xlsx', 'E:E');
% total_prediction_360 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\single-point prediction online\prediction_360_4800_1600_1600.xlsx', 'F:F');
% test_prediction_360 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\single-point prediction online\prediction_360_4800_1600_1600.xlsx', 'G:G');
% train_prediction_pre_360 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\single-point prediction online\prediction_360_4800_1600_1600.xlsx', 'H:H');
% validation_prediction_pre_360 = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\single-point prediction online\prediction_360_4800_1600_1600.xlsx', 'I:I');
% real_total_360 = load('channel_data_360_10.mat');
% real_total_360 = real_total_360.h;
% real_total_360 = real_total_360(1,1:8000);
% real_total_360 = abs(real_total_360);
% 
% 
% train_begin = 1;
% train_end = 4800;
% validation_end = 6400;
% train_num_pre = 200;

%% 总预测图
figure(1)
subplot(221)
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
plot([4800, 4800], [-0.20, 1.2],'r', 'LineWidth', 1.25 )
plot([6400, 6400], [-0.20, 1.2],'r', 'LineWidth', 1.25 )
grid on
ylabel('|h_{2}(t)|, v=180 km/h')

%% 总误差图
subplot(222)
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
subplot(223)
t_real_total_360 = 1:8000;
plot(t_real_total_360, real_total_360,'LineWidth', 1.25);
hold on
t_train_prediction_pre_360 = 6:train_end;
plot(t_train_prediction_pre_360, train_prediction_pre_360, 'LineWidth', 1.25);
hold on
t_validation_prediction_pre_360 = train_end+1:validation_end;
plot(t_validation_prediction_pre_360, validation_prediction_pre_360, 'LineWidth', 1.25);
hold on
t_test_prediction_360 = validation_end+1:8000;
plot(t_test_prediction_360, test_prediction_360, 'LineWidth', 1.25)
hold on
t_prediction_total_360 = 6:8000;
plot(t_prediction_total_360, total_prediction_360, 'LineWidth', 1.25);
hold on
legend('Real channel data', 'Prediction based on train data by pre-training model', 'Prediction based on validation data by pre-training model', 'Online prediction by IL', 'Prediction based on total data by the IL model when time-slot is 8000');
plot([4800, 4800], [-0.20, 1.4],'r', 'LineWidth', 1.25 )
plot([6400, 6400], [-0.20, 1.4],'r', 'LineWidth', 1.25 )
grid on
xlabel('Time-slot')
ylabel('|h_{3}(t)|, v=360 km/h')

%% 总误差图
subplot(224)
% error_train_pre = zeros(1, train_end-5);
% error_validation_pre = zeros(1:validation_end-train_end);
t_error_train_pre_360 = 6:train_end;
t_error_validation_pre_360 = train_end+1:validation_end;
t_error_test_IL_360 = validation_end+1:8000;
t_error_total_360 = 6:8000;
error_train_pre_360 = train_prediction_pre_360 - real_total_360(6: train_end)';
error_validation_pre_360 = validation_prediction_pre_360 - real_total_360(train_end+1:validation_end)';
error_test_IL_360 = test_prediction_360 - real_total_360(6401:8000)';
error_total_360 = total_prediction_360 - real_total_360(6:8000)';
scatter(t_error_train_pre_360, error_train_pre_360, '.');
hold on 
scatter(t_error_validation_pre_360, error_validation_pre_360, '.');
hold on 
scatter(t_error_test_IL_360, error_test_IL_360, '.');
hold on 
scatter(t_error_total_360, error_total_360, '.');
hold on 
plot([1, 8000],[0, 0],'c-', 'LineWidth', 1.75)
hold on
legend('Error based on train data', 'Error based on validation data','Online prediction error', 'Error based total data by by the IL model when time-slot is 8000', 'Zero-error reference');
grid on
xlabel('Time-slot')
ylabel('Error')


% 
% 
% 
% %% 总loss图
% train_loss_pre = train_loss;
% figure(3)
% t_train_loss_pre = 1:1799;
% semilogy(t_train_loss_pre, train_loss_pre, 'LineWidth', 1.25);
% hold on
% t_valadation_loss = 1:200;
% semilogy(t_valadation_loss, validation_loss, 'LineWidth', 1.25);
% hold on
% legend('Train loss', 'Validation loss');
% plot([200, 200], [10^(-7), 1],'r', 'LineWidth', 1.25 )
% grid on
% xlabel('Pre-training and IL step')
% ylabel('Loss')
