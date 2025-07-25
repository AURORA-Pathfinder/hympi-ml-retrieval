import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from sklearn.cluster import AgglomerativeClustering
from scipy.interpolate import interp1d
import netCDF4 as nc
from sklearn.linear_model import Ridge , LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split


def read_netcdf(filename):
 ncf =  nc.Dataset(filename)
 Freq=ncf.variables['FREQUENCY'][:]
 TB=ncf.variables['BT'][:]
 return Freq,TB

def run_cluster(data_TB,indices, threshold,neighborhood,nedt=None):
    sub_data = data_TB[indices,:]
     
    if nedt is not None:
#        nedt_sub = nedt[indices]
        sub_data = sub_data / nedt  # Normalize by NEDT per channel
        sub_data = sub_data - sub_data.mean(axis=0)  # Remove mean (zero-mean per channel)


    corr = np.corrcoef(sub_data)
    print(corr.shape)


    dist = 1 - np.abs(corr)

    # Build connectivity for just this region
    conn = lil_matrix((len(indices), len(indices)))
    for i in range(len(indices)):
        for j in range(1, neighborhood + 1):
            if i - j >= 0:
                conn[i, i - j] = 1
            if i + j < len(indices):
                conn[i, i + j] = 1

    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        connectivity=conn,
        distance_threshold=threshold,
        n_clusters=None
    )

    local_labels = clustering.fit_predict(dist)
    return local_labels



def compute_center_bandwidth(n_groups,frequencies,labels,TB):

 group_centers = []
 group_bandwidths = []
 group_counts = []
 group_freqs =[]
 group_TBs=[] 
 
 channel_spacing = frequencies[1]-frequencies[0] #0.003906

 for group_id in range(n_groups):
    group_freq = frequencies[labels == group_id]


    group_TB=TB[labels == group_id,:]  
    counts = len(group_freq)
    if counts == 1:
        center = group_freq[0]
        bandwidth = channel_spacing

    elif counts % 2 == 0:
        center = group_freq[counts // 2 - 1]
        bandwidth = group_freq[-1] - group_freq[0]
    else:
        center = group_freq[counts // 2]
        bandwidth = group_freq[-1] - group_freq[0]

    group_centers.append(center)
    group_bandwidths.append(bandwidth)
    group_counts.append(counts)
    group_TBs.append(group_TB)
    group_freqs.append(group_freq)

# --- SORT BY CENTER FREQUENCY ---
 group_info = sorted(zip(group_centers, group_bandwidths,group_counts,group_TBs,group_freqs), key=lambda x: x[0])
 group_centers_sorted, group_bandwidths_sorted ,group_counts_sorted,group_TBs_sorted,group_freqs_sorted= zip(*group_info)

 group_centers_sorted = list(group_centers_sorted)
 group_bandwidths_sorted = list(group_bandwidths_sorted)
 group_counts_sorted=list(group_counts_sorted)
 group_freqs_sorted=list(group_freqs_sorted)
 group_TBs_sorted=list(group_TBs_sorted)
 return group_centers_sorted,group_bandwidths_sorted,group_counts_sorted,group_freqs_sorted,group_TBs_sorted

def bar_plot(img_name,group_centers_sorted,group_counts_sorted,group_bandwidths_sorted):
 plt.figure(figsize=(12, 4))
 plt.bar(
    group_centers_sorted,
    group_counts_sorted,     
    width=group_bandwidths_sorted,
    align='center',
    edgecolor='k')

 plt.xlabel("Center Frequency (GHz)")
 plt.ylabel("Channel Count")
 plt.title("Grouped Channel Bandwidths")
 plt.grid(True, axis='y', linestyle='--', alpha=0.5)
 plt.tight_layout()
 plt.savefig(img_name)

class GroupedSRFProcessor:
    def __init__(self, groups):
        self.groups = groups
        self.center_freqs = []
        self.avg_tbs = []  # List of arrays, each (n_profiles,) for a group

    @staticmethod
    def trapezoidal_rule(values, x):
        return np.trapz(values, x)

    def process(self):
        for group in self.groups:
            freqs = np.array(group['freqs'])              # shape: (n_channels,)
            tbs = np.array(group['tbs'])                  # shape: (n_channels, n_profiles)
            bw = group['bandwidth']
            cf = group['center_freq']

            # Ensure tbs is 2D even for single-profile case
            if tbs.ndim == 1:
                tbs = tbs[:, np.newaxis]  # shape becomes (n_channels, 1)

            # Bandwidth-based masking
            f_start = cf - bw / 2
            f_end = cf + bw / 2
            mask = (freqs >= f_start) & (freqs <= f_end)

            selected_freqs = freqs[mask]                  # (n_selected,)
            selected_tbs = tbs[mask, :]                   # (n_selected, n_profiles)

            if len(selected_freqs) < 2:

                avg_tb = selected_tbs[0, :]
            else:
                numerator = np.array([
                    self.trapezoidal_rule(selected_tbs[:, p], selected_freqs)
                    for p in range(tbs.shape[1])
                ])
                denominator = self.trapezoidal_rule(np.ones_like(selected_freqs), selected_freqs)
                avg_tb = numerator / denominator if denominator != 0 else np.full(tbs.shape[1], np.nan)

            self.center_freqs.append(cf)
            self.avg_tbs.append(avg_tb)  # shape: (n_profiles,)



def Filter_Data(TB_all,frequencies,n_groups):

 TB_filtered = np.zeros_like(TB_all)
 F_filtered=np.zeros((n_groups))

 compressed_data = np.zeros_like(TB_all)  # Same shape as original TB data (n_profiles, n_channels)
 filt_freq=np.zeros_like(frequencies)

 for group_id in range(n_groups):
    group_indices = np.where(labels == group_id)[0]

    # Find center index as before
    group_freqs_sorted = np.sort(group_indices)
    if len(group_freqs_sorted) % 2 == 1:
        center_idx = group_freqs_sorted[len(group_freqs_sorted) // 2]
    else:
        center_idx = group_freqs_sorted[len(group_freqs_sorted) // 2 - 1]

    # For all profiles, assign only the center TB, rest stay zero
    compressed_data[:, center_idx] = TB_all[:, center_idx]
    filt_freq[center_idx]=frequencies[center_idx]

 nonZeroIndices = (filt_freq != 0)

 TB_filtered = TB_all[:, nonZeroIndices]
 F_filtered  = filt_freq[nonZeroIndices]
 return TB_filtered,F_filtered

def recontruct_interp(TB_filtered,F_filtered,frequencies):

 if np.ma.isMaskedArray(frequencies):
    frequencies = frequencies.filled()

 interp_func = interp1d(F_filtered.squeeze(),TB_filtered.squeeze(),kind='nearest', bounds_error=False,fill_value='extrapolate')
 NewTB_all = interp_func(frequencies.squeeze())
 return NewTB_all


def reconstruct_regression_train_test(compressed_data, data_TB, alpha=1.0, test_size=0.2, random_state=42):
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        compressed_data, data_TB, test_size=test_size, random_state=random_state)
    
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)  # Train only on training data

    reconstructed_test = model.predict(compressed_data)  # Predict on unseen test data

    return reconstructed_test#, y_test, model


def compute_stats(TB_all,NewTB_all,group_centers_sorted):


 error_bias=np.mean(TB_all-NewTB_all,axis=0)
 error_stdv=np.std(TB_all-NewTB_all,axis=0)

 NewTBavg=np.mean(NewTB_all,axis=0)
 TBavg_all=np.mean(TB_all,axis=0)

 nobs=TB_all.shape[0]
 nch=TB_all.shape[1]
 ngroup=np.shape(group_centers_sorted)[0] 
 
 nedt_all = np.full(nch, 6.4)
 tau=0.010 #seconds
 Tant=290   #K
 Trec=1000   #1000 for 180GHZ 450 for 50GHZ
 Tsys=Tant+Trec
 group_freq=np.array(group_centers_sorted)
 Bsub=np.diff(group_freq)
 Bsub=np.append(Bsub, Bsub[-1]) * 1000000000
 Nedt_sub = Tsys / (np.sqrt(Bsub * tau))


 #variance
 ss_res = np.sum((TB_all - NewTB_all) ** 2, axis=0)
 ss_tot = np.sum((TB_all - np.mean(TB_all, axis=0)) ** 2, axis=0)
 r_squared = 1 - (ss_res / ss_tot)
 var_exp = 100 * r_squared * ((nch * nobs - 1) / (nch * nobs - ngroup - 1))


 print(ngroup,nch,nobs)
 print(var_exp.min(),var_exp.max())
 print(error_bias.min(),error_bias.max())
 print(error_stdv.min(),error_stdv.max())
 return error_bias,error_stdv,NewTBavg,TBavg_all,Nedt_sub,nedt_all,var_exp,group_freq


def plot_stats(frequencies,group_freq,TBavg_all,NewTBavg,error_bias,var_exp,error_stdv,nedt_all,Nedt_sub,imgname):

 freq=frequencies
 sns.set_style('darkgrid')

 plt.rcParams.update({'font.size': 18})

 fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4,ncols=1,figsize=(12, 16),dpi=300)

 ax1.plot(freq, TBavg_all,color='red', linewidth=2.0, label='Truth')
 ax1.plot(freq, NewTBavg,color='blue', linewidth=2.0, linestyle='--',label='Reconstructed')

 ax1.set_title('TBs',  fontsize=18)
 ax1.legend(fontsize=16,loc='upper left')
 ax1.grid(True)

 # Mean Bias
 ax2.plot(freq, error_bias, color='blue', linewidth=2.0, label='MeanBias')
 ax2.set_title('Mean Bias', fontsize=18)
 ax2.set_ylim(-0.2, 0.2)
 #ax2.set_ylabel('Error (K)', fontsize=14)
 ax2.grid(True)

# Variance Explained
 ax3.plot(freq, var_exp, color='blue', linewidth=2.0)
 ax3.set_title('Variance Explained', fontsize=18)
 #ax3.set_ylabel('Error (K)', fontsize=14)
 ax3.grid(True)
# ax3.set_ylim(0, 100)
# Standard Deviation + NEDT
 ax4.plot(freq, error_stdv, color='blue', linewidth=2.0, label='Std Dev')
 ax4.plot(freq, nedt_all, 'g--', linewidth=2.0, label='NEDT All')
 ax4.plot(group_freq, Nedt_sub, 'm--', linewidth=2.0, label='NEDT Grouped')
 ax4.set_title('Standard Deviation & NEDT', fontsize=18)
 ax4.set_xlabel('Frequency', fontsize=18)
 #ax4.set_ylabel('Error (K)', fontsize=14)
 #ax4.legend()
 ax4.legend(fontsize=16,loc='upper left') 
 ax4.grid(True)
 ax4.set_ylim(0, 6.5)
 
# Use tight_layout to minimize overlapping
 plt.tight_layout()
 
 # Save the figure with a high resolution (dpi=300)
 plt.savefig(imgname,dpi=300,
            bbox_inches='tight',
            pad_inches=0.1)






 
filename='/discover/nobackup/nshahrou/RT_spectral/Convolve_monoRTM/new_TB_dec_183GHZ_3p9.nc'
img_path='./newimages2/'
test='newtest17'



frequencies1,TB_all1=read_netcdf(filename)

frequencies=frequencies1[502:]
TB_all=TB_all1[:,502:]

data_TB = TB_all.T  
n_channels=data_TB.shape[0]
n_profiless=data_TB.shape[1]

print(data_TB.shape)






TB = (data_TB / 6.4) - data_TB.mean(axis=1).reshape(-1, 1)
print(TB.min(),TB.max())
print(data_TB.min(),data_TB.max())

R = np.corrcoef(TB)
#plt.figure(figsize=(10, 8))
#plt.rcParams.update({'font.size': 18})
#plt.imshow(R, cmap='jet')
#plt.colorbar()
#num_ticks = 17
#tickrange = np.linspace(0, n_channels, num_ticks, dtype=int)  # Positions on the axis
#frequency_range = np.linspace(175, 191, num_ticks)        # Corresponding labels
#plt.xticks(tickrange,frequency_range, rotation=90)
#plt.yticks(tickrange,frequency_range)
#plt.savefig(img_path+test+'_corr_matrix_cosmir-h2',bbox_inches='tight',pad_inches = 0.1)




# Split and Combine
#low_indices = np.where(frequencies <  181)[0]
#mid_indices = np.where((frequencies >= 181) & (frequencies < 185))[0]
#high_indices = np.where(frequencies >= 185)[0]

low_indices = np.where(frequencies <  178)[0]
mid_indices1 = np.where((frequencies >= 178) & (frequencies < 182))[0]
mid_indices2 = np.where((frequencies >= 182) & (frequencies < 184))[0]
mid_indices3 = np.where((frequencies >= 184) & (frequencies < 188))[0]
high_indices = np.where(frequencies >= 188)[0]



labels_low = run_cluster(data_TB,low_indices, threshold=0.00001,neighborhood = 5,nedt=6.4)
labels_mid1 = run_cluster(data_TB,mid_indices1, threshold=0.001,neighborhood = 5,nedt=6.4)
labels_mid2 = run_cluster(data_TB,mid_indices2, threshold=0.0001,neighborhood = 2,nedt=6.4)
labels_mid3 = run_cluster(data_TB,mid_indices3, threshold=0.001,neighborhood = 5,nedt=6.4)
labels_high= run_cluster(data_TB,high_indices,threshold=0.0001,neighborhood = 5,nedt=6.4)
labels = np.full(n_channels, -1, dtype=int)

#labels[low_indices]  = labels_low
#offset_mid = labels_low.max() + 1
#labels[mid_indices]  = labels_mid + offset_mid
#offset_high = labels[mid_indices].max() + 1
#labels[high_indices] = labels_high + offset_high
#n_groups = labels.max() + 1


labels[low_indices] = labels_low
offset = labels_low.max() + 1
labels[mid_indices1] = labels_mid1 + offset
offset = labels[mid_indices1].max() + 1
labels[mid_indices2] = labels_mid2 + offset
offset = labels[mid_indices2].max() + 1
labels[mid_indices3] = labels_mid3 + offset
offset = labels[mid_indices3].max() + 1
labels[high_indices] = labels_high + offset
n_groups = labels.max() + 1



print(f"Total groups: {n_groups}")
#print(data_TB.shape)

#compute center,count,bandwidth 
group_centers_sorted,group_bandwidths_sorted,group_counts_sorted,group_freqs_sorted,group_TBs_sorted=compute_center_bandwidth(n_groups,frequencies,labels,data_TB)

beginFreq=np.zeros(n_groups)
endFreq=np.zeros(n_groups)

for n in range(n_groups):   
 F=group_freqs_sorted[n]
 beginFreq[n]=F[0]
 endFreq[n]=F[-1]



# print(group_counts_sorted[n],group_centers_sorted[n],group_bandwidths_sorted[n],group_freqs_sorted[n],group_TBs_sorted[n][:, 0])

groups = []
for n in range(n_groups):
    group_dict = {
        'freqs': group_freqs_sorted[n].tolist(),
        'tbs': group_TBs_sorted[n].tolist(),  # using profile 0
        'bandwidth': group_bandwidths_sorted[n],
        'center_freq': group_centers_sorted[n]
    }

    groups.append(group_dict)


processor = GroupedSRFProcessor(groups)
processor.process()


TB_filtered = np.vstack(processor.avg_tbs).T           # shape: (n_groups, n_profiles)
F_filtered = np.array(processor.center_freqs)     # shape: (n_groups,)



#print(TB_filtered.shape)
#print(tb_array[:,0])
#print(freq_array)


#for p in range(tb_array.shape[1]):
#    plt.plot(freq_array, tb_array[:, p], marker='o', label=f'Profile {p}')
#
#plt.xlabel("Center Frequency (GHz)")
#plt.ylabel("Average TB (K)")
#plt.title("Avg TB vs Center Frequency")
##plt.grid(True)
#plt.legend()
#plt.tight_layout()
#plt.show()

#quit()
img_name=img_path+test+f"_bar_183GHZ_group_{n_groups}"
#bar_plot(img_name,group_centers_sorted,group_counts_sorted,group_bandwidths_sorted)








#TB_filtered2,F_filtered=Filter_Data(TB_all,frequencies,n_groups)
#print(TB_filtered2.shape)


#NewTB_all_interp =  recontruct_interp(TB_filtered,F_filtered,frequencies)

NewTB_all_regress = reconstruct_regression_train_test(TB_filtered, TB_all, alpha=1000, test_size=0.8)



#print(NewTB_all_interp.shape,group_centers_sorted)
#print(np.shape(group_centers_sorted))
#quit()

#error_bias1,error_stdv1,NewTBavg1,TBavg_all1,Nedt_sub1,nedt_all1,var_exp1,group_freq1=compute_stats(TB_all,NewTB_all_interp,group_centers_sorted)
error_bias2,error_stdv2,NewTBavg2,TBavg_all2,Nedt_sub2,nedt_all2,var_exp2,group_freq2=compute_stats(TB_all,NewTB_all_regress,group_centers_sorted)



#imgname1=img_path+test+f"_interp_comsir_h2_{n_groups}"
#plot_stats(frequencies,group_freq1,TBavg_all1,NewTBavg1,error_bias1,var_exp1,error_stdv1,nedt_all1,Nedt_sub1,imgname1)
imgname2=img_path+test+f"_regress_cosmir-h2_{n_groups}"
#plot_stats(frequencies,group_freq2,TBavg_all2,NewTBavg2,error_bias2,var_exp2,error_stdv2,nedt_all2,Nedt_sub2,imgname2)

np.savez(test+"_grouped_TB_183_data.npz", TB=TB_filtered, freq=F_filtered,nedt=Nedt_sub2,bandwidth=group_bandwidths_sorted,beginFreq=beginFreq,endFreq=endFreq)


quit()
