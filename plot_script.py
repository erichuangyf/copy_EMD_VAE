import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 8)
from sample_and_analysis import VAE_sampler


original_data_sampler = VAE_sampler("/global/home/users/yifengh3/new_data/B_background.h5",
                     "/global/home/users/yifengh3/VAE/B_results/method2_beta2",
                    train_valid_split=800000)

b_signal_sampler = VAE_sampler("/global/home/users/yifengh3/new_data/B_signal.h5",
                     "/global/home/users/yifengh3/VAE/B_results/method2_beta2")

h_signal_sampler = VAE_sampler("/global/home/users/yifengh3/new_data/h_signal.h5",
                     "/global/home/users/yifengh3/VAE/B_results/method2_beta2")

hv_signal_sampler = VAE_sampler("/global/home/users/yifengh3/new_data/hv_signal.h5",
                     "/global/home/users/yifengh3/VAE/B_results/method2_beta2")


plotting_method_name = \
            [method for method in dir(VAE_sampler) if method.startswith('_VAE_sampler__plots')]
plotting_function = [eval("VAE_sampler."+func_name) for func_name in plotting_method_name]
background_output = original_data_sampler.y[:,:,:3]
b_output = b_signal_sampler.y[:,:,:3]
h_output = h_signal_sampler.y[:,:,:3]
hv_output =  hv_signal_sampler.y[:,:,:3]
print(b_output.shape)
call_args = [background_output, b_output, "/global/home/users/yifengh3/new_data/data_plots"]
call_kargs = {"additional_signal":[h_output,hv_output], "data_name":["Background","B Jets","H Jets","HV Jets"]}
for count,func in enumerate(plotting_function):
    plt.figure()
    print("getting plot {} out of {}".format(count+1,len(plotting_function)))
    func(*call_args,**call_kargs)
    plt.show()



