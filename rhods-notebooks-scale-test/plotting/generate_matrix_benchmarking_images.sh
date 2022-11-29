export MATBENCH_PLOTTING_IMAGE_WIDTH=700
export MATBENCH_PLOTTING_IMAGE_HEIGHT=500

export MATBENCH_RESULTS_DIRNAME=../000__driver_rhods__notebook_osd_ci_scale_test
export MATBENCH_WORKLOAD=rhods-notebooks-ux

mkdir -p out
rm out/* -rf

matbench visualize --generate 'stats=Notebook spawn time&cfg=hide_launch_delay=y'
mv fig_{00,01}_Notebook_spawn_time.html
mv fig_{00,01}_Notebook_spawn_time.png

matbench visualize --generate "\
stats=report: Error report&\
stats=report: Master Nodes Load&\
stats=Notebook spawn time&\
stats=Execution time distribution&cfg=time_to_reach_step=Go to JupyterLab Page&\
stats=Prom: Sutest Master Node CPU idle&\
stats=Prom: RHODS Dashboard: CPU usage"

mv fig_* report_* out

montage -geometry +1+1 dashboard/*  out/dashboard.png
