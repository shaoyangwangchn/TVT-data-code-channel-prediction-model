% clc;
% clear;
% train_loss = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_16.xlsx', 'A:A');
% validation_loss = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_16.xlsx', 'B:B');
% test_loss = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_16.xlsx', 'C:C');
% train_prediction = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_16.xlsx', 'D:D');
% validation_prediction = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_16.xlsx', 'E:E');
% total_prediction = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_16.xlsx', 'F:F');
% test_prediction = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_16.xlsx', 'G:G');
% train_prediction_pre = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_16.xlsx', 'H:H');
% validation_prediction_pre = xlsread('E:\PHD\0论文\0论文\TVT_2018\code\time series prediction_CSDN\multi-point prediction online\prediction_90_length_16.xlsx', 'I:I');
% real_total = load('channel_data_90_10.mat');
% real_total = real_total.h;
% real_total = real_total(1,1:8000);
% real_total = abs(real_total);
% 
% train_begin = 1;
% train_end = 4800;
% validation_end = 6400;
% train_num_pre = 200;
% %% 预训练阶段的预测图
% real_pre = real_total(1:6400);
% figure(1)
% t_real_pre = 1:6400;
% % plot(t_real_pre, real_pre, 'b', 'LineWidth', 1.25);
% plot(t_real_pre, real_pre, 'LineWidth', 1.25);
% hold on
% t_train_prediction_pre = 6:train_end;
% % plot(t_train_prediction_pre, train_prediction_pre,'--','Color', [1 0.84 0], 'LineWidth', 1.25);
% % plot(t_train_prediction_pre, train_prediction_pre,'-','Color', [0.93 0.69 0.13], 'LineWidth', 1.25);
% plot(t_train_prediction_pre, train_prediction_pre, 'LineWidth', 1.25);
% hold on
% t_validation_prediction_pre = train_end+1:validation_end;
% % plot(t_validation_prediction_pre, validation_prediction_pre,'g-', 'LineWidth', 1.25);
% plot(t_validation_prediction_pre, validation_prediction_pre, 'LineWidth', 1.25);
% hold on
% legend('Real channel data', 'Prediction based on train data', 'Prediction based on validation data')
% plot([4800, 4800], [0, 1],'r', 'LineWidth', 1.25 )
% grid on
% xlabel('Time-slot, v=90 km/h')
% ylabel('|h_{1}(t)|')
% 
% %% 预训练阶段Loss图(直角坐标系)
% train_loss_pre = train_loss(1:train_num_pre);
% figure(2)
% t_train_loss_pre = 1:200;
% plot(t_train_loss_pre, train_loss_pre, 'LineWidth', 1.25);
% hold on
% plot(t_train_loss_pre, validation_loss, 'LineWidth', 1.25);
% hold on
% legend('Train loss', 'Validation loss');
% grid on
% xlabel('Pre-training step')
% ylabel('Loss')
% 
% %% 预训练阶段Loss图(直角坐标系)
% train_loss_pre = train_loss(1:train_num_pre);
% figure(3)
% t_train_loss_pre = 1:200;
% semilogy(t_train_loss_pre, train_loss_pre, 'LineWidth', 1.25);
% hold on
% semilogy(t_train_loss_pre, validation_loss, 'LineWidth', 1.25);
% hold on
% legend('Train loss', 'Validation loss');
% grid on
% xlabel('Pre-training step')
% ylabel('Loss')
% 
% %% 误差图
% % error_train_pre = zeros(1, train_end-5);
% % error_validation_pre = zeros(1:validation_end-train_end);
% t_error_train_pre = 1:train_end-5;
% t_error_validation_pre = train_end-4:validation_end-5;
% error_train_pre = train_prediction_pre - real_pre(6: train_end)';
% error_validation_pre = validation_prediction_pre - real_pre(train_end+1:validation_end)';
% figure(4)
% scatter(t_error_train_pre, error_train_pre, '.');
% hold on 
% scatter(t_error_validation_pre, error_validation_pre, '.');
% hold on 
% plot([1, validation_end-5],[0, 0],'c-', 'LineWidth', 1.75)
% hold on
% legend('Error based on train data', 'Error based on validation data', 'Zero-error reference');
% grid on
% xlabel('Time-slot')
% ylabel('Error')
% 
% %% 增量学习阶段的预测图
% figure(5)
% real_IL = real_total(validation_end+1:8000);
% t_real_IL = 1:1600;
% plot(t_real_IL, real_IL,'LineWidth', 1.25);
% hold on
% t_test_prediction = 1:1600;
% plot(t_test_prediction, test_prediction,'LineWidth', 1.25);
% hold on
% legend('Real channel data', 'Online prediction');
% grid on
% xlabel('Time-slot, v=90 km/h')
% ylabel('|h_{1}(t)|')
% 
% %% 增量学习阶段的误差图
% figure(6)
% error_IL = test_prediction - real_IL';
% t_error_IL = 1:1600;
% scatter(t_error_IL, error_validation_pre, '.');
% hold on 
% plot([1, 1600],[0, 0],'c-', 'LineWidth', 1.75)
% hold on
% legend('Prediction error', 'Zero-error reference');
% grid on
% xlabel('Time-slot')
% ylabel('Error')
% 
% %% 增量学习Loss
% figure(7)
% train_loss_IL = train_loss(201:1799);
% t_train_IL = 1:(1799-200);
% semilogy(t_train_IL, train_loss_IL, 'LineWidth', 1.25);
% hold on
% legend('IL loss');
% grid on
% xlabel('IL step')
% ylabel('Loss')

%% 总预测图
figure(8)
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
plot([4800, 4800], [-0.2, 1],'r', 'LineWidth', 1.25 )
plot([6400, 6400], [-0.2, 1],'r', 'LineWidth', 1.25 )
grid on
xlabel('Time-slot, v=90 km/h')
ylabel('|h_{1}(t)|')

%% 总误差图
figure(9)
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
xlabel('Time-slot')
ylabel('Error')

%% 总loss图
train_loss_pre = train_loss;
figure(10)
t_train_loss_pre = 1:299;
semilogy(t_train_loss_pre, train_loss_pre, 'LineWidth', 1.25);
hold on
t_valadation_loss = 1:200;
semilogy(t_valadation_loss, validation_loss, 'LineWidth', 1.25);
hold on
legend('Train loss', 'Validation loss');
plot([200, 200], [10^(-5), 1],'r', 'LineWidth', 1.25 )
grid on
xlabel('Pre-training and IL step')
ylabel('Loss')


