clear
CNN_baseline = [82.79 
92.25 
75.21 
82.63 
86.50 
96.58 
78.08 
88.92 
82.42 
90.50 
83.67 
70.96 
91.33 
83.96 
84.04 
82.13 

];



SANet = [94.08
95.79
89.96
94.67
92.5
96.13
87.79
93.88
91.25
91.67
96.38
91
92.46
85.92
89.67
95.29
];

TANet = [94.33
95.58
89.54
93.71
92
94.71
88.63
93.25
93.46
91.29
96.33
92.92
91.04
88.71
96.67
93.79
];

STANet = [97.42
96.13
87.29
95.25
92.42
95.04
89.62
91.75
92.71
91.08
97.13
94
92.79
88.54
92.29
96
];


[h1, p1, ci1, stats1] = ttest(SANet,CNN_baseline);
[h2, p2, ci2, stats2] = ttest(TANet,SANet);
[h3, p3, ci3, stats3] = ttest(STANet,TANet);

data = [CNN_baseline SANet TANet STANet];
mean_data = mean(data, 1);


set(groot, 'defaultAxesFontName','Times New Roman');
set(groot, 'defaultTextFontName','Times New Roman');
set(groot, 'defaultAxesFontSize', 14);
set(groot, 'defaultTextFontSize', 14);

figure
xlim([0.5, 4.5])
ylim([70, 104])
set(gca, 'FontWeight', 'bold');
hold on
box on;
load("col.mat")

for i = 1:size(data, 1)
    line_color = col(:,i)';
    plot(data(i, :), 'Color', line_color, 'LineStyle', '--', 'LineWidth', 1.5, 'Marker', 'o', 'MarkerSize', 3, 'MarkerFaceColor', line_color, 'Color', [line_color,0.6])
end


plot(mean_data, '-o', 'LineWidth', 1.8, 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', 'Color', 'k')


sem_data = std(data)/sqrt(size(data,1));
errorbar(mean_data, sem_data, 'LineStyle', 'none', 'LineWidth', 1.5, 'Color', 'k')



set(gca, 'XTick', 1:4)
set(gca, 'XTickLabel', {'CNN-baseline','SANet', 'TANet', 'STANet'})
xlabel('Model')
ylabel('Decoding accuracy(%)')
title('')

