set term pngcairo size 1536,1024 linewidth 2 fontscale 2;

set title "Random environments";
set xlabel "Number of gradient updates";
set ylabel "Average regret";
set log y;
set grid x;
set grid y;
set key box;

set output "random1.png";
plot \
    "1_dr.csv" u 12:23 w l title " DR";

set output "random2.png";
plot \
    "1_dr.csv" u 12:23 w l title " DR", \
    "2_plr_abs.csv" u 4:19 w l title "PLR^⊥";

set output "random3.png";
plot \
    "1_dr.csv" u 12:23 w l title " DR", \
    "2_plr_abs.csv" u 4:19 w l title "PLR^⊥", \
    "3_plr_norm.csv" u 7:32 w l title "PLR^⊥ (score normalised)";

set output "random4.png";
plot \
    "1_dr.csv" u 12:23 w l title " DR", \
    "2_plr_abs.csv" u 4:19 w l title "PLR^⊥", \
    "3_plr_norm.csv" u 7:32 w l title "PLR^⊥ (score normalised)", \
    "4_accel.csv" u 7:32 w l title "ACCEL";


set title "Wall environments";

set output "walls1.png";
plot \
    "1_dr.csv" u 12:34 w l title "DR";

set output "walls2.png";
plot \
    "1_dr.csv" u 12:34 w l title "DR", \
    "2_plr_abs.csv" u 4:30 w l title "PLR^⊥";

set output "walls3.png";
plot \
    "1_dr.csv" u 12:34 w l title "DR", \
    "2_plr_abs.csv" u 4:30 w l title "PLR^⊥", \
    "3_plr_norm.csv" u 7:43 w l title "PLR^⊥ (score normalised)";

set output "walls4.png";
plot \
    "1_dr.csv" u 12:34 w l title "DR", \
    "2_plr_abs.csv" u 4:30 w l title "PLR^⊥", \
    "3_plr_norm.csv" u 7:43 w l title "PLR^⊥ (score normalised)", \
    "4_accel.csv" u 7:43 w l title "ACCEL";
