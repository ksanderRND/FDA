function dataset = preprocess_data(name)

Table = readtable(name);
[N,M] = size(Table);

unique_jobs = table2cell(unique(Table(:, 2)));
unique_marital = table2cell(unique(Table(:, 3)));
unique_education = table2cell(unique(Table(:, 4)));
unique_default = table2cell(unique(Table(:, 5)));
unique_housing = table2cell(unique(Table(:, 7)));
unique_loan = table2cell(unique(Table(:, 8)));
unique_contact = table2cell(unique(Table(:, 9)));
unique_month = table2cell(unique(Table(:, 11)));
unique_poutcome = table2cell(unique(Table(:, 16)));
unique_y = table2cell(unique(Table(:, 17)));
num_of_params = [0 1 12 3 4 2 1 2 2 3 1 12 1 1 1 1 4 2];
num_of_columns = cumsum(num_of_params);
    
dataset = zeros(num_of_columns(end)-1, N);
data = zeros(num_of_columns(end), N);
Values = table2cell(Table(:, 1:17));

for i=1:N
    for j=1:length(num_of_params)-1
        switch j
            case 2
                unique_values=unique_jobs;
            case 3
                unique_values=unique_marital;
            case 4
                unique_values=unique_education;
            case 5
                unique_values=unique_default;
            case 7
                unique_values=unique_housing;
            case 8
                unique_values=unique_loan;
            case 9
                unique_values=unique_contact;
            case 11
                unique_values=unique_month;
            case 16
                unique_values=unique_poutcome;
            case 17
                unique_values=unique_y;
            otherwise
                unique_values = 0;
        end
        if (num_of_params(j+1) == 1)
            data(num_of_columns(j+1), i) = Values{i, j};
        else
            index = find(ismember(unique_values, Values{i, j}));
            data(num_of_columns(j) + index, i) = 1;
        end
    end
 end
 
dataset = (minmaxnorm(data',0,1))';
end