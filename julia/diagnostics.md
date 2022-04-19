# Diagnostics for julia optimisation

Base problem, auto-alg, 100x100 (1e-8,1e3):

 ─────────────────────────────────────────────────────────────────────────────────────────
                                                 Time                    Allocations      
                                        ───────────────────────   ────────────────────────
            Tot / % measured:                83.7s /  94.2%           9.73GiB /  94.7%    

 Section                        ncalls     time    %tot     avg     alloc    %tot      avg
 ─────────────────────────────────────────────────────────────────────────────────────────
 complete co run                     1    78.8s  100.0%   78.8s   9.21GiB  100.0%  9.21GiB
   main part co                      1    53.0s   67.3%   53.0s   5.90GiB   64.0%  5.90GiB
     problem definition co           1    28.7s   36.4%   28.7s   2.66GiB   28.8%  2.66GiB
     solving co                  10.0k    16.7s   21.2%  1.67ms   2.44GiB   26.5%   256KiB
     solving 1st co                  1    6.59s    8.4%   6.59s    659MiB    7.0%   659MiB
     problem re-definition co    10.0k   43.8ms    0.1%  4.38μs   5.48MiB    0.1%     575B
   co network def                    1    4.53s    5.7%   4.53s    393MiB    4.2%   393MiB
 ─────────────────────────────────────────────────────────────────────────────────────────

Rosenbrock23(), 100x100 (1e-8, 1e3):

─────────────────────────────────────────────────────────────────────────────────────────
                                                 Time                    Allocations      
                                        ───────────────────────   ────────────────────────
            Tot / % measured:                79.6s /  93.6%           9.07GiB /  94.3%    

 Section                        ncalls     time    %tot     avg     alloc    %tot      avg
 ─────────────────────────────────────────────────────────────────────────────────────────
 complete co run                     1    74.6s  100.0%   74.6s   8.56GiB  100.0%  8.56GiB
   main part co                      1    47.8s   64.2%   47.8s   5.24GiB   61.2%  5.24GiB
     problem definition co           1    29.7s   39.8%   29.7s   2.66GiB   31.0%  2.66GiB
     solving co                  10.0k    8.76s   11.8%   876μs   1.22GiB   14.2%   127KiB
     solving 1st co                  1    8.41s   11.3%   8.41s   1.22GiB   14.2%  1.22GiB
     problem re-definition co    10.0k   44.8ms    0.1%  4.48μs   5.48MiB    0.1%     574B
   co network def                    1    4.74s    6.4%   4.74s    393MiB    4.5%   393MiB
 ─────────────────────────────────────────────────────────────────────────────────────────

Auto-alg, 100x100, (1e2, 1e3)

─────────────────────────────────────────────────────────────────────────────────────────
                                                 Time                    Allocations      
                                        ───────────────────────   ────────────────────────
            Tot / % measured:                82.8s /  94.2%           9.70GiB /  94.7%    

 Section                        ncalls     time    %tot     avg     alloc    %tot      avg
 ─────────────────────────────────────────────────────────────────────────────────────────
 complete co run                     1    78.0s  100.0%   78.0s   9.19GiB  100.0%  9.19GiB
   main part co                      1    51.2s   65.7%   51.2s   5.87GiB   63.9%  5.87GiB
     problem definition co           1    27.1s   34.7%   27.1s   2.66GiB   28.9%  2.66GiB
     solving co                  10.0k    16.5s   21.1%  1.65ms   2.41GiB   26.3%   253KiB
     solving 1st co                  1    6.65s    8.5%   6.65s    659MiB    7.0%   659MiB
     problem re-definition co    10.0k   44.3ms    0.1%  4.43μs   5.48MiB    0.1%     575B
   setup                             1    5.17s    6.6%   5.17s    393MiB    4.2%   393MiB
     co network def                  1    5.17s    6.6%   5.17s    393MiB    4.2%   393MiB
 ─────────────────────────────────────────────────────────────────────────────────────────

Auto-alg, 100x100, (1e2, 1e3) ring vs co:

 ───────────────────────────────────────────────────────────────────────────────────────────
                                                   Time                    Allocations      
                                          ───────────────────────   ────────────────────────
             Tot / % measured:                 88.5s /  94.7%           10.5GiB /  95.1%    

 Section                          ncalls     time    %tot     avg     alloc    %tot      avg
 ───────────────────────────────────────────────────────────────────────────────────────────
 complete ring remake                  1    55.5s   66.1%   55.5s   6.57GiB   65.8%  6.57GiB
   main ring remake                    1    32.2s   38.4%   32.2s   3.43GiB   34.3%  3.43GiB
     problem definition ring           1    25.7s   30.7%   25.7s   2.56GiB   25.6%  2.56GiB
     solving 1st ring                  1    5.92s    7.1%   5.92s    574MiB    5.6%   574MiB
     solving ring                  10.0k    415ms    0.5%  41.5μs    307MiB    3.0%  31.5KiB
     problem re-definition ring    10.0k   68.9ms    0.1%  6.89μs   5.43MiB    0.1%     570B
   ring network def                    1    3.29s    3.9%   3.29s    268MiB    2.6%   268MiB
 complete co run                       1    28.4s   33.9%   28.4s   3.42GiB   34.2%  3.42GiB
   main part co                        1    26.3s   31.3%   26.3s   3.23GiB   32.4%  3.23GiB
     solving co                    10.0k    16.6s   19.8%  1.66ms   2.41GiB   24.2%   253KiB
     problem definition co             1    4.34s    5.2%   4.34s    359MiB    3.5%   359MiB
     solving 1st co                    1    4.29s    5.1%   4.29s    328MiB    3.2%   328MiB
     calc n                            1    657ms    0.8%   657ms   63.9MiB    0.6%  63.9MiB
     recalc n                      10.0k    269ms    0.3%  26.9μs   76.8MiB    0.8%  7.86KiB
     problem re-definition co      10.0k   46.8ms    0.1%  4.68μs   5.43MiB    0.1%     570B
   setup                               1    892ms    1.1%   892ms    100MiB    1.0%   100MiB
     co network def                    1    891ms    1.1%   891ms    100MiB    1.0%   100MiB
 ───────────────────────────────────────────────────────────────────────────────────────────

Auto-alg, 1000x1000, (1e2, 1e3):

─────────────────────────────────────────────────────────────────────────────────────────
                                                 Time                    Allocations      
                                        ───────────────────────   ────────────────────────
            Tot / % measured:                1769s /  99.7%            257GiB /  99.8%    

 Section                        ncalls     time    %tot     avg     alloc    %tot      avg
 ─────────────────────────────────────────────────────────────────────────────────────────
 complete co run                     1    1764s  100.0%   1764s    257GiB  100.0%   257GiB
   main part co                      1    1739s   98.6%   1739s    253GiB   98.7%   253GiB
     solving co                  1.00M    1664s   94.3%  1.66ms    241GiB   94.1%   253KiB
     recalc n                    1.00M    29.3s    1.7%  29.3μs   7.50GiB    2.9%  7.86KiB
     problem definition co           1    27.0s    1.5%   27.0s   2.66GiB    1.0%  2.66GiB
     solving 1st co                  1    6.60s    0.4%   6.60s    659MiB    0.3%   659MiB
     problem re-definition co    1.00M    1.12s    0.1%  1.12μs    353MiB    0.1%     370B
     calc n                          1    612ms    0.0%   612ms   63.9MiB    0.0%  63.9MiB
   setup                             1    4.41s    0.2%   4.41s    393MiB    0.1%   393MiB
     co network def                  1    4.41s    0.2%   4.41s    393MiB    0.1%   393MiB
 ─────────────────────────────────────────────────────────────────────────────────────────

Rodas5, 100x100, (1e2, 1e3)

─────────────────────────────────────────────────────────────────────────────────────────
                                                 Time                    Allocations      
                                        ───────────────────────   ────────────────────────
            Tot / % measured:                77.2s /  93.5%           8.59GiB /  94.0%    

 Section                        ncalls     time    %tot     avg     alloc    %tot      avg
 ─────────────────────────────────────────────────────────────────────────────────────────
 complete co run                     1    72.2s  100.0%   72.2s   8.08GiB  100.0%  8.08GiB
   main part co                      1    45.1s   62.5%   45.1s   4.75GiB   58.9%  4.75GiB
     problem definition co           1    27.5s   38.0%   27.5s   2.66GiB   32.9%  2.66GiB
     solving 1st co                  1    9.16s   12.7%   9.16s   1.28GiB   15.9%  1.28GiB
     solving co                  10.0k    7.46s   10.3%   746μs    682MiB    8.2%  69.8KiB
     calc n                          1    618ms    0.9%   618ms   63.9MiB    0.8%  63.9MiB
     recalc n                    10.0k    243ms    0.3%  24.3μs   76.8MiB    0.9%  7.86KiB
     problem re-definition co    10.0k   43.5ms    0.1%  4.35μs   5.48MiB    0.1%     574B
   setup                             1    4.56s    6.3%   4.56s    393MiB    4.8%   393MiB
     co network def                  1    4.56s    6.3%   4.56s    393MiB    4.8%   393MiB
 ─────────────────────────────────────────────────────────────────────────────────────────

Rosenbrock23, 100x100, (1e2, 1e3)

─────────────────────────────────────────────────────────────────────────────────────────
                                                 Time                    Allocations      
                                        ───────────────────────   ────────────────────────
            Tot / % measured:                74.5s /  93.5%           9.06GiB /  94.3%    

 Section                        ncalls     time    %tot     avg     alloc    %tot      avg
 ─────────────────────────────────────────────────────────────────────────────────────────
 complete co run                     1    69.6s  100.0%   69.6s   8.55GiB  100.0%  8.55GiB
   main part co                      1    44.4s   63.7%   44.4s   5.22GiB   61.1%  5.22GiB
     problem definition co           1    27.0s   38.8%   27.0s   2.66GiB   31.1%  2.66GiB
     solving co                  10.0k    8.33s   12.0%   833μs   1.20GiB   14.0%   126KiB
     solving 1st co                  1    8.04s   11.5%   8.04s   1.22GiB   14.2%  1.22GiB
     calc n                          1    611ms    0.9%   611ms   63.9MiB    0.7%  63.9MiB
     recalc n                    10.0k    240ms    0.3%  24.0μs   76.8MiB    0.9%  7.86KiB
     problem re-definition co    10.0k   44.0ms    0.1%  4.40μs   5.48MiB    0.1%     574B
   setup                             1    4.39s    6.3%   4.39s    393MiB    4.5%   393MiB
     co network def                  1    4.39s    6.3%   4.39s    393MiB    4.5%   393MiB
 ─────────────────────────────────────────────────────────────────────────────────────────

Steady-state Rodas5() 1000x1000:

─────────────────────────────────────────────────────────────────────────────────────────
                                                 Time                    Allocations      
                                        ───────────────────────   ────────────────────────
            Tot / % measured:                1128s /  99.6%           58.7GiB /  99.1%    

 Section                        ncalls     time    %tot     avg     alloc    %tot      avg
 ─────────────────────────────────────────────────────────────────────────────────────────
 complete co steady-state            1    1124s  100.0%   1124s   58.2GiB  100.0%  58.2GiB
   main part co                      1    1102s   98.0%   1102s   55.4GiB   95.1%  55.4GiB
     solving co                  1.00M    1028s   91.5%  1.03ms   42.9GiB   73.7%  45.0KiB
     recalc n                    1.00M    25.9s    2.3%  25.9μs   7.50GiB   12.9%  7.86KiB
     problem definition co           1    23.8s    2.1%   23.8s   2.32GiB    4.0%  2.32GiB
     solving 1st co                  1    12.4s    1.1%   12.4s   1.59GiB    2.7%  1.59GiB
     problem re-definition co    1.00M    2.09s    0.2%  2.09μs    355MiB    0.6%     372B
     calc n                          1    1.04s    0.1%   1.04s    108MiB    0.2%   108MiB
   setup                             1    4.59s    0.4%   4.59s    394MiB    0.7%   394MiB
     co network def                  1    4.59s    0.4%   4.59s    394MiB    0.7%   394MiB
 ─────────────────────────────────────────────────────────────────────────────────────────

Steady-state Rodas5() 1719x1719:


 ─────────────────────────────────────────────────────────────────────────────────────────
                                                 Time                    Allocations      
                                        ───────────────────────   ────────────────────────
            Tot / % measured:                3319s /  99.8%            158GiB /  99.7%    

 Section                        ncalls     time    %tot     avg     alloc    %tot      avg
 ─────────────────────────────────────────────────────────────────────────────────────────
 complete co steady-state            1    3314s  100.0%   3314s    158GiB  100.0%   158GiB
   main part co                      1    3290s   99.3%   3290s    155GiB   98.1%   155GiB
     solving co                  2.94M    3132s   94.5%  1.07ms    126GiB   79.8%  45.0KiB
     recalc n                    2.94M    82.1s    2.5%  27.9μs   22.0GiB   13.9%  7.86KiB
     problem definition co           1    23.9s    0.7%   23.9s   2.30GiB    1.5%  2.30GiB
     solving 1st co                  1    12.5s    0.4%   12.5s   1.57GiB    1.0%  1.57GiB
     problem re-definition co    2.94M    7.37s    0.2%  2.51μs   1.01GiB    0.6%     369B
     calc n                          1    1.08s    0.0%   1.08s    111MiB    0.1%   111MiB
   setup                             1    4.54s    0.1%   4.54s    399MiB    0.2%   399MiB
     co network def                  1    4.54s    0.1%   4.54s    399MiB    0.2%   399MiB
 ─────────────────────────────────────────────────────────────────────────────────────────

Jacobian (1e-8,1e6) Rodas5() 10x10 with abstraction (serial):

 ─────────────────────────────────────────────────────────────────────────────────
                                         Time                    Allocations      
                                ───────────────────────   ────────────────────────
        Tot / % measured:            91.6s /  52.8%           9.28GiB /  46.8%    

 Section                ncalls     time    %tot     avg     alloc    %tot      avg
 ─────────────────────────────────────────────────────────────────────────────────
 complete                    1    43.5s   90.0%   43.5s   3.99GiB   91.8%  3.99GiB
   main run                  1    42.9s   88.7%   42.9s   3.94GiB   90.8%  3.94GiB
     make problem            1    14.7s   30.3%   14.7s   1.29GiB   29.6%  1.29GiB
     make jac problem        1    10.7s   22.2%   10.7s    907MiB   20.4%   907MiB
     1st solve               1    8.66s   17.9%   8.66s   1.10GiB   25.2%  1.10GiB
     create jacobian         1    7.00s   14.5%   7.00s    515MiB   11.6%   515MiB
     solve                 100    100ms    0.2%   997μs   6.53MiB    0.1%  66.8KiB
     remake problem        100   19.3μs    0.0%   193ns     0.00B    0.0%    0.00B
 co network def              1    4.86s   10.0%   4.86s    364MiB    8.2%   364MiB
 ─────────────────────────────────────────────────────────────────────────────────

Jacobian (1e-8,1e6) Rodas5() 100x100 with abstraction (serial & parallel):

 ────────────────────────────────────────────────────────────────────────────────
                                        Time                    Allocations      
                               ───────────────────────   ────────────────────────
       Tot / % measured:             124s /  96.5%           13.2GiB /  96.2%    

 Section               ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────────────────
 initial compilation        1    90.9s   75.9%   90.9s   9.68GiB   76.3%  9.68GiB
   main run                 2    41.8s   34.8%   20.9s   3.98GiB   31.4%  1.99GiB
   co network def           2    4.42s    3.7%   2.21s    364MiB    2.8%   182MiB
 complete serial            2    20.7s   17.3%   10.3s   1.50GiB   11.9%   771MiB
   main run                 2    20.7s   17.3%   10.3s   1.50GiB   11.9%   770MiB
 complete parallel          2    8.18s    6.8%   4.09s   1.50GiB   11.8%   770MiB
   main run                 2    8.18s    6.8%   4.09s   1.50GiB   11.8%   769MiB
 co network def             4   27.3ms    0.0%  6.83ms   1.02MiB    0.0%   262KiB
 ────────────────────────────────────────────────────────────────────────────────

Jacobian (1e-8,1e6) Rodas5() 1719x1719 with abstraction (serial & parallel):

 ────────────────────────────────────────────────────────────────────────────────
                                        Time                    Allocations      
                               ───────────────────────   ────────────────────────
       Tot / % measured:            4350s /  99.9%            445GiB /  99.9%    

 Section               ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────────────────
 complete serial            1    3067s   70.6%   3067s    218GiB   48.9%   218GiB
   main run                 1    3067s   70.6%   3067s    218GiB   48.9%   218GiB
 complete parallel          1    1186s   27.3%   1186s    217GiB   48.9%   217GiB
   main run                 1    1186s   27.3%   1186s    217GiB   48.9%   217GiB
 initial compilation        1    93.2s    2.1%   93.2s   9.68GiB    2.2%  9.68GiB
   main run                 2    41.8s    1.0%   20.9s   3.98GiB    0.9%  1.99GiB
   co network def           2    4.46s    0.1%   2.23s    364MiB    0.1%   182MiB
 co network def             2   2.00ms    0.0%  1.00ms    524KiB    0.0%   262KiB
 ────────────────────────────────────────────────────────────────────────────────

Jacobian (1e4,1e6) Rodas5() 1719x1719 with abstraction (serial & parallel):

────────────────────────────────────────────────────────────────────────────────
                                        Time                    Allocations      
                               ───────────────────────   ────────────────────────
       Tot / % measured:            4391s /  99.9%            445GiB /  99.9%    

 Section               ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────────────────
 complete serial            1    3086s   70.4%   3086s    217GiB   48.9%   217GiB
   main run                 1    3086s   70.4%   3086s    217GiB   48.9%   217GiB
 complete parallel          1    1210s   27.6%   1210s    217GiB   48.9%   217GiB
   main run                 1    1209s   27.6%   1209s    217GiB   48.9%   217GiB
 initial compilation        1    91.1s    2.1%   91.1s   9.68GiB    2.2%  9.68GiB
   main run                 2    41.4s    0.9%   20.7s   3.98GiB    0.9%  1.99GiB
   co network def           2    4.39s    0.1%   2.20s    364MiB    0.1%   182MiB
 co network def             2   2.02ms    0.0%  1.01ms    524KiB    0.0%   262KiB
 ────────────────────────────────────────────────────────────────────────────────
