training_data = importdata('train_data.m');
validation_data = importdata('validation_data.m');

training_input = training_data(:,1:2);
training_target = training_data(:,3);

validation_input = validation_data(:,1:2);
validation_target = validation_data(:,3);

class_one = (training_input(find(training_target == 1),:));
class_two = (training_input(find(training_target == -1),:));
size(class_one)
size(class_two)

hold on
scatter(class_one(:,1), class_one(:,2), 'b');
scatter(class_two(:,1), class_two(:,2), 'r');