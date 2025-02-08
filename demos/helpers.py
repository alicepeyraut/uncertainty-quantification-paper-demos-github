import numpy
import os


def write_porosity(porosity_field = [], n_cells = 0, filepath = "./"):
    print("filepath", filepath)
    with open(filepath, "w") as file:
        file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        file.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
        file.write('  <mesh_function type="double" dim="'+str(3)+'" size="'+str(n_cells)+'">\n')
        for k_cell in range(n_cells):
            file.write('    <entity index="'+str(k_cell)+'" value="'+str(porosity_field[k_cell])+'"/>\n')
        file.write('  </mesh_function>\n')
        file.write('</dolfin>\n')
        file.close()


def checking_if_processes_converged(processes):
    if not processes:  # Check if the list is empty
        print("Warning: No processes found.")
        return True 
    over = True
    for process in processes:
        if process.poll() is None:  # If any process is still running
            over = False  # Not all processes are done
            break 
    return(over)


def checking_if_converged(params_opt = {}, nb_processes_converged = 0, tol = 1e-3):
    converged = False ### if distribution is converged, converged becomes True
    if nb_processes_converged == 0: ### no process converged
        print("Warning, no process converged")
        return(converged, None)
    error_old = 0 ### needed because there can be more than one list
    for list_ in params_opt:
        lst = list_[1:] ### the first item of the list corresponds to the parameter name
        if abs(numpy.std(lst)) > 45: ### the standard deviation of the error distribution is wider than the input distribution; the distribution will not converge
            return(converged, None)
        median = abs((numpy.percentile(lst[:], 50) - numpy.percentile(lst[:-nb_processes_converged], 50))/(numpy.percentile(lst[:-nb_processes_converged], 50) if numpy.percentile(lst[:-nb_processes_converged], 50) !=0 else 1))
        q1 = abs((numpy.percentile(lst[:], 25)-numpy.percentile(lst[:-nb_processes_converged], 25))/(numpy.percentile(lst[:-nb_processes_converged], 25) if numpy.percentile(lst[:-nb_processes_converged], 25) !=0 else 1))
        q3 = abs((numpy.percentile(lst[:], 75)-numpy.percentile(lst[:-nb_processes_converged], 75))/(numpy.percentile(lst[:-nb_processes_converged], 75) if numpy.percentile(lst[:-nb_processes_converged], 75) !=0 else 1))
        std = abs((numpy.std(lst[:]) - numpy.std(lst[:-nb_processes_converged]))/(numpy.std(lst[:-nb_processes_converged]) if numpy.std(lst[:-nb_processes_converged]) !=0 else 1))
        error_max = max(median, q1, q3, std, error_old)
        error_old = error_max
    if error_max < float(tol):
        converged = True
    return(converged, error_max)


def initialize_directories(current_directory):
    directory_prone = current_directory + "/prone" ### storing the results created for prone position
    directory_supine = current_directory + "/supine" ### storing the results created for supine position
    ### creating associated directories, if do not already exist
    try:
        os.mkdir(directory_prone)
    except:
        pass
    try:
        os.mkdir(directory_supine)
    except:
        pass
    return(directory_prone, directory_supine)


def initialize_lsts_and_params(parameters_to_identify, noise_or_bias_lst, noise_or_bias):
    nb_parameters = len(parameters_to_identify)
    reference_value = []
    param_names = []
    for param_name, param_value in parameters_to_identify.items():
        reference_value.append(param_value)
        param_names.append(param_name)
    storing_processes = {} ### storing processes launched in parallel
    results = {} ### storing the results of the estimation, for each noise level
    storing_values_for_convergence_check = {} ### storing the results for convergence checks
    ### initializing lists in results dict
    if str(noise_or_bias) == 'noise':
        results['noise'] = []
    else:
        results['bias'] = []
    for param_name in param_names:
        lst_name = "ini_"+str(param_name)
        results[lst_name] = []
        results[param_name] = []
    for noise in noise_or_bias_lst:
        storing_processes[str(noise)] = []
        storing_values_for_convergence_check[str(noise)] = []
        for i in range(nb_parameters):
            storing_values_for_convergence_check[str(noise)].append([])
    return(results, storing_values_for_convergence_check, storing_processes, reference_value, param_names, nb_parameters)