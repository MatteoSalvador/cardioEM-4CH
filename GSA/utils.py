import numpy as np

output_labels = ["pLA_min", "pLA_max", "dpLA_dt_min", "dpLA_dt_max",
                 "VLA_min", "VLA_max", "dVLA_dt_min", "dVLA_dt_max",
                 "pLV_min", "pLV_max", "dpLV_dt_min", "dpLV_dt_max",
                 "VLV_min", "VLV_max", "dVLV_dt_min", "dVLV_dt_max",
                 "pRA_min", "pRA_max", "dpRA_dt_min", "dpRA_dt_max",
                 "VRA_min", "VRA_max", "dVRA_dt_min", "dVRA_dt_max",
                 "pRV_min", "pRV_max", "dpRV_dt_min", "dpRV_dt_max",
                 "VRV_min", "VRV_max", "dVRV_dt_min", "dVRV_dt_max"]

params_names = ["PCa_b", "trpnmax", "Gncx_b",
                "Tref_V", "perm50_V", "nperm_V",
                "TRPN_n_V", "dr_V", "wfrac_V",
                "TOT_A_V", "ktm_unblock", "ca50_V",
                "mu_V", "maxI_up", "maxTrpn",
                "g_CaL", "Tref_A", "perm50_A",
                "nperm_A", "TRPN_n_A", "dr_A",
                "wfrac_A", "TOT_A_A", "phi",
                "ca50_A", "mu_A", "CV_ventricles",
                "k_FEC", "CV_atria", "k_BB",
                "AV_delay", "Rsys", "Rpulm","Aol",
                "kArt", "a_ventricles", "bt_ventricles",
                "a_atria", "bf_atria", "bt_atria",
                "k_peri", "a_lvrv", "Tref_lvrv"]
 
params_bounds = [[6.36596000e-05, 1.53905000e-04],
                 [6.47620000e-02, 1.22782000e-01],
                 [9.34194000e-04, 2.42994000e-03],
                 [1.27846000e+02, 1.99575000e+02],
                 [1.76440000e-01, 5.11652000e-01],
                 [1.85418000e+00, 3.04409000e+00],
                 [1.83896000e+00, 2.99802000e+00],
                 [1.26338000e-01, 3.62711000e-01],
                 [2.88399000e-01, 7.46166000e-01],
                 [1.26888000e+01, 3.73920000e+01],
                 [1.23413000e-02, 3.14574000e-02],
                 [4.07102000e-01, 1.04899000e+00],
                 [1.52163000e+00, 4.48439000e+00],
                 [2.76308000e-03, 7.97911000e-03],
                 [3.58952000e-02, 1.03881000e-01],
                 [8.91029000e-02, 1.97851000e-01],
                 [8.01142000e+01, 1.19934000e+02],
                 [1.80217000e-01, 5.23327000e-01],
                 [2.54302000e+00, 7.38619000e+00],
                 [1.02219000e+00, 2.98411000e+00],
                 [1.27122000e-01, 3.71089000e-01],
                 [2.61181000e-01, 7.46106000e-01],
                 [1.27021000e+01, 3.71894000e+01],
                 [1.17375000e+00, 3.43049000e+00],
                 [5.67801000e-01, 1.28798000e+00],
                 [4.60653000e+00, 1.33936000e+01],
                 [3.83230000e-01, 7.96718000e-01],
                 [1.32498000e+00, 8.36869000e+00],
                 [7.50815000e-01, 1.02690000e+00],
                 [1.70108000e+00, 5.63717000e+00],
                 [1.00000000e+02, 1.99000000e+02],
                 [1.00166000e+00, 3.99368000e+00],
                 [1.00196000e+00, 3.99797000e+00],
                 [3.00478000e+02, 4.98745000e+02],
                 [6.01180000e+00, 9.98937000e+00],
                 [5.00607000e-01, 1.49978000e+00],
                 [1.50492000e+00, 4.49251000e+00],
                 [1.50951000e+00, 2.49991000e+00],
                 [4.04925000e+00, 1.19966000e+01],
                 [1.51919000e+00, 4.49892000e+00],
                 [5.05981000e-04, 1.99910000e-03],
                 [1.00548000e+00, 1.99947000e+00],
                 [5.00917000e-01, 9.95545000e-01]]
n_params = len(params_names) # 43

def get_problem():
    params = np.array(params_bounds)

    problem = {
        'num_vars': n_params,
        'names': params_names,
        'bounds': params
    }

    return problem
