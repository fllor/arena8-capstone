set term pngcairo size 1536,1024 linewidth 2 fontscale 2;
set output "walls.png";

set title "Wall environments";
set xlabel "Number of gradient updates";
set ylabel "Average regret";
set log y;
set grid x;
set grid y;
set key box;

plot \
    "1_dr.csv" u 12:34 w l title "DR", \
    "2_plr_abs.csv" u 4:30 w l title "PLR^⊥", \
    "3_plr_norm.csv" u 7:43 w l title "PLR^⊥ (score normalised)", \
    "4_accel.csv" u 7:43 w l title "ACCEL";
