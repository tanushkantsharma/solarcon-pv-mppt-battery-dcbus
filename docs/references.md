# References

1. M. G. Villalva, J. R. Gazoli, E. Ruppert Filho, "Comprehensive approach to modeling and simulation of photovoltaic arrays," *IEEE Trans. Power Electronics*, 24(5), 1198–1208, 2009. — single-diode modelling, `a = 1.3` choice, `I0(T)` form.
2. Kyocera, *KC200GT High Efficiency Multicrystal Photovoltaic Module* datasheet. — STC and NOCT electrical data used for parameter extraction and validation.
3. A. Jain, A. Kapoor, "Exact analytical solutions of the parameters of real solar cells using Lambert W-function," *Solar Energy Materials and Solar Cells*, 81(2), 269–277, 2004. — explicit I(V) solution.
4. N. Femia, G. Petrone, G. Spagnuolo, M. Vitelli, "Optimization of perturb and observe maximum power point tracking method," *IEEE Trans. Power Electronics*, 20(4), 963–973, 2005. — P&O step/period design.
5. K. H. Hussein, I. Muta, T. Hoshino, M. Osakada, "Maximum photovoltaic power tracking: an algorithm for rapidly changing atmospheric conditions," *IEE Proc. Generation, Transmission and Distribution*, 142(1), 59–64, 1995. — incremental conductance.
6. D. Sera, T. Kerekes, R. Teodorescu, F. Blaabjerg, "Improved MPPT method for rapidly changing environmental conditions," *Proc. IEEE ISIE 2006*, 1420–1425, doi:10.1109/ISIE.2006.295680. — dP-P&O drift compensation.
7. F. Liu, S. Duan, F. Liu, B. Liu, Y. Kang, "A variable step size INC MPPT method for PV systems," *IEEE Trans. Industrial Electronics*, 55(7), 2622–2628, 2008. — variable-step principle.
8. T. Esram, P. L. Chapman, "Comparison of photovoltaic array maximum power point tracking techniques," *IEEE Trans. Energy Conversion*, 22(2), 439–449, 2007. — MPPT survey.
9. R. W. Erickson, D. Maksimović, *Fundamentals of Power Electronics*, 3rd ed., Springer, 2020. — state-space averaging, CCM converter models.
10. G. L. Plett, *Battery Management Systems, Vol. I: Battery Modeling*, Artech House, 2015. — equivalent-circuit (Thevenin) battery models.
11. K. J. Åström, T. Hägglund, *Advanced PID Control*, ISA, 2006. — back-calculation anti-windup.
12. EN 50530:2010, *Overall efficiency of grid connected photovoltaic inverters*. — context for dynamic MPPT test ramps.

Reference 1's citation details and reference 6 (including DOI) and the KC200GT datasheet values were checked online while building this project; the remaining entries are standard textbook/journal references cited from the author's knowledge and should be checked against the publisher record before formal submission.
