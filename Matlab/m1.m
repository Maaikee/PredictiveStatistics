generate_figs = true;

GT1 = readtable('gt_2011.csv');
GT2 = readtable('gt_2012.csv');
GT3 = readtable('gt_2013.csv');
GT4 = readtable('gt_2014.csv');
GT5 = readtable('gt_2015.csv');

GT1A = table2array(GT1);
GT1A(:,10) = [];
GT2A = table2array(GT2);
GT2A(:,10) = [];
GT3A = table2array(GT3);
GT3A(:,10) = [];
GT4A = table2array(GT4);
GT4A(:,10) = [];
GT5A = table2array(GT5);
GT5A(:,10) = [];

% means
column_max = 11;
means_raw(1,1) = 2011;
means_raw(1, 2:column_max) = mean(GT1A, 1);
means_raw(2,1) = 2012;
means_raw(2, 2:column_max) = mean(GT2A, 1);
means_raw(3,1) = 2013;
means_raw(3, 2:column_max) = mean(GT3A, 1);
means_raw(4,1) = 2014;
means_raw(4, 2:column_max) = mean(GT4A, 1);
means_raw(5,1) = 2015;
means_raw(5, 2:column_max) = mean(GT5A, 1);

means = array2table(means_raw, 'VariableNames', {'YEAR','AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'NOX'});
if generate_figs
    figure(1);
    years = [2011 2012 2013 2014 2015];
    plot(means_raw(:, 2:column_max));
    title('Means (2011-2015)');
    legend('AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'NOX');
    xlabel('Year');
    ylabel('Value')
    grid;
end
writetable(means, 'means.csv');
    
% medians 
median1 = median(GT1A, 1);
percentile1 = prctile(GT1A, 1, 1);
percentile99 = prctile(GT1A, 99, 1);

stddev_raw(1,1) = 2011;
stddev_raw(1, 2:column_max) = std(GT1A, 1);
stddev_raw(2,1) = 2012;
stddev_raw(2, 2:column_max) = std(GT2A, 1);
stddev_raw(3,1) = 2013;
stddev_raw(3, 2:column_max) = std(GT3A, 1);
stddev_raw(4,1) = 2014;
stddev_raw(4, 2:column_max) = std(GT4A, 1);
stddev_raw(5,1) = 2015;
stddev_raw(5, 2:column_max) = std(GT5A, 1);

stddev = array2table(stddev_raw, 'VariableNames', {'YEAR','AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'NOX'});
if generate_figs
    figure(2);
    plot(stddev_raw(1:5, 2:column_max));
    title('Standard deviation (2011-2015)');
    legend('AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'NOX');
    xlabel('Year');
    ylabel('Value')
    grid;
end

range_raw(1,1) = 2011;
range_raw(1, 2:column_max) = range(GT1A);
range_raw(2,1) = 2012;
range_raw(2, 2:column_max) = range(GT2A);
range_raw(3,1) = 2013;
range_raw(3, 2:column_max) = range(GT3A);
range_raw(4,1) = 2014;
range_raw(4, 2:column_max) = range(GT4A);
range_raw(5,1) = 2015;
range_raw(5, 2:column_max) = range(GT5A);

ranges = array2table(range_raw, 'VariableNames', {'YEAR','AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'NOX'});
if generate_figs
    figure(3);
    plot(range_raw(1:5, 2:column_max));
    title('Ranges (2011-2015)');
    legend('AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'NOX');
    xlabel('Year');
    ylabel('Value')
    grid;
end

rowNames = {'AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'NOX'};

corr2011_raw = corr(GT1A);
corr2011 = array2table(corr2011_raw, 'VariableNames', rowNames);
corr2011.Properties.RowNames = rowNames;
writetable(corr2011, 'corr2011.csv');

corr2012_raw = corr(GT2A);
corr2012 = array2table(corr2012_raw, 'VariableNames', rowNames);
corr2012.Properties.RowNames = rowNames;
writetable(corr2012, 'corr2012.csv');

corr2013_raw = corr(GT3A);
corr2013 = array2table(corr2013_raw, 'VariableNames', rowNames);
corr2013.Properties.RowNames = rowNames;
writetable(corr2013, 'corr2013.csv');

corr2014_raw = corr(GT4A);
corr2014 = array2table(corr2014_raw, 'VariableNames', rowNames);
corr2014.Properties.RowNames = rowNames;
writetable(corr2014, 'corr2014.csv');

corr2015_raw = corr(GT5A);
corr2015 = array2table(corr2015_raw, 'VariableNames', rowNames);
corr2015.Properties.RowNames = rowNames;
writetable(corr2015, 'corr2015.csv');

% Plot correlations between features and NOx over time.
corrNOx_raw(1,2) = 2011;
corrNOx_raw(2:column_max, 2) = corr2011_raw(:, 10);
corrNOx_raw(1,3) = 2012;
corrNOx_raw(2:column_max, 3) = corr2012_raw(:, 10);
corrNOx_raw(1,4) = 2013;
corrNOx_raw(2:column_max, 4) = corr2013_raw(:, 10);
corrNOx_raw(1,5) = 2014;
corrNOx_raw(2:column_max, 5) = corr2014_raw(:, 10);
corrNOx_raw(1,6) = 2015;
corrNOx_raw(2:column_max, 6) = corr2015_raw(:, 10);

corrNOx = array2table(corrNOx_raw);
writetable(corrNOx, 'corrNOx_2011-2015.csv');

figure(4);
t_corrNOx_raw = transpose(corrNOx_raw);
plot(t_corrNOx_raw(2:6, 2:column_max));
set(gca,'ydir','reverse')
title('Correlations NOx over time (2011-2015)');
legend('AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'NOX');
xlabel('Year');
ylabel('Value')
grid;

% TODO: Detect > 0.2 absolute change between correlation tables 
diff_corr_raw = corr2015_raw - corr2011_raw;
diff_corr = array2table(diff_corr_raw, 'VariableNames', rowNames);
diff_corr.Properties.RowNames = rowNames;
writetable(diff_corr, 'corr_diff_2011_and_2015.csv');

% TODO: Detect top 3 variables whose correlations with other variables
% changed the most from 2011 to 2015 (only use those 2)


% write tables
table2011 = evaluate_data(GT1A);
writetable(table2011, 'stats2011.csv');
table2012 = evaluate_data(GT2A);
writetable(table2012, 'stats2012.csv');
table2013 = evaluate_data(GT3A);
writetable(table2013, 'stats2013.csv');
table2014 = evaluate_data(GT4A);
writetable(table2014, 'stats2014.csv');
table2015 = evaluate_data(GT5A);
writetable(table2015, 'stats2015.csv');

overall = [GT1A;GT2A;GT3A;GT4A;GT5A];
table_overall = evaluate_data(overall);
writetable(table_overall, 'stats_overall.csv');

function table = evaluate_data(file)
    m(1, 2:11) = mean(file, 1);
    m(2, 2:11) = median(file, 1);
    m(3, 2:11) = prctile(file, 1, 1);
    m(4, 2:11) = prctile(file, 99, 1);
    m(5, 2:11) = std(file, 1);
    m(6, 2:11) = range(file);
    
    table = array2table(m, 'VariableNames', {'VALUE','AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'TEY', 'CDP', 'NOX'});
end
