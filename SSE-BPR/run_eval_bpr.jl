function compute_pairwise_error_ndcg(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k, cold_start_s)
	sum_error = 0.; ndcg_sum = 0.;
	for i in collect(cold_start_s)
		tmp = nzrange(Y, i)
		d2_bar = rows_t[tmp];
		vals_d2_bar = vals_t[tmp];
		ui = U[:, i]
		len = size(d2_bar)[1]
		score = zeros(len)
		for j = 1:len
			J = d2_bar[j];
			vj = V[:, J]
			score[j] = dot(ui,vj)
		end
		error_this = 0
		n_comps_this = 0
		for j in 1:(len - 1)
			jval = vals_d2_bar[j]
			for k in (j + 1):len
				kval = vals_d2_bar[k]
				if score[j] >= score[k] && jval < kval
					error_this += 1
				end
				if score[j] <= score[k] && jval > kval
					error_this += 1
				end
				n_comps_this += 1
			end
		end
		sum_error += error_this / n_comps_this


		p1 = sortperm(score, rev = true)
		p1 = p1[1:ndcg_k]
		M1 = vals_d2_bar[p1]
		p2 = sortperm(vals_d2_bar, rev = true)
		p2 = p2[1:ndcg_k]
		M2 = vals_d2_bar[p2]
		dcg = 0.; dcg_max = 0.
		for k = 1:ndcg_k
			dcg += (2 ^ M1[k] - 1) / log2(k + 1)
			dcg_max += (2 ^ M2[k] - 1) / log2(k + 1)
		end
		ndcg_sum += dcg / dcg_max
	end
	return sum_error / length(cold_start_s), ndcg_sum / length(cold_start_s)
end

function compute_pairwise_error_ndcg(U, V, Y, r, d1, d2, rows_t, vals_t, cols_t, ndcg_k)
	sum_error = 0.; ndcg_sum = 0.;
	for i = 1:d1
		tmp = nzrange(Y, i)
		d2_bar = rows_t[tmp];
		vals_d2_bar = vals_t[tmp];
		ui = U[:, i]
		len = size(d2_bar)[1]
		score = zeros(len)
		for j = 1:len
			J = d2_bar[j];
			vj = V[:, J]
			score[j] = dot(ui,vj)
		end
		error_this = 0
		n_comps_this = 0
		for j in 1:(len - 1)
			jval = vals_d2_bar[j]
			for k in (j + 1):len
				kval = vals_d2_bar[k]
				if score[j] >= score[k] && jval < kval
					error_this += 1
				end
				if score[j] <= score[k] && jval > kval
					error_this += 1
				end
				n_comps_this += 1
			end
		end
		sum_error += error_this / n_comps_this


		p1 = sortperm(score, rev = true)
		p1 = p1[1:ndcg_k]
		M1 = vals_d2_bar[p1]
		p2 = sortperm(vals_d2_bar, rev = true)
		p2 = p2[1:ndcg_k]
		M2 = vals_d2_bar[p2]
		dcg = 0.; dcg_max = 0.
		for k = 1:ndcg_k
			dcg += (2 ^ M1[k] - 1) / log2(k + 1)
			dcg_max += (2 ^ M2[k] - 1) / log2(k + 1)
		end
		ndcg_sum += dcg / dcg_max
	end
	return sum_error / d1, ndcg_sum / d1
end



function compute_precision(U, V, X, Y, d1, d2, rows, vals, rows_t, vals_t, zero_items)
	K = [1, 5, 10] # K has to be increasing order
	precision = [0, 0, 0]
	for i = 1:d1
    #for i = shuffle(1:d1)[1:500]
		tmp = nzrange(Y, i)
		test = Set{Int64}()
		for j in tmp 
			push!(test, rows_t[j])
		end
		test = Set(rows_t[tmp])
		if isempty(test)
			continue
		end
		tmp = nzrange(X, i)
		vals_d2_bar = vals[tmp]
		train = Set(rows[tmp])
		score = zeros(d2)
		ui = U[:, i]
		for j = 1:d2
            #if j in train || !(j in zero_items)
			if j in train	
                score[j] = -10e10
				continue
			end
			vj = V[:, j]
			score[j] = dot(ui,vj)
		end
		p = sortperm(score, rev = true)
		for c = 1: K[length(K)]
			j = p[c]
			if score[j] == -10e10
				break
			end
			if j in test
				for k in length(K):-1:1
					if c <= K[k]
						precision[k] += 1
					else
						break
					end
				end
			end
		end
	end
	precision = precision./K/d1
	#precision = precision./K/500
    return precision[1], precision[2], precision[3]
end

train = "ml1m-50-oc/ml1m_oc_50_train_ratings.csv"
test = "ml1m-50-oc/ml1m_oc_50_test_ratings.csv"
X = readdlm(train, ',' , Int64);
x = vec(X[:,1]);
y = vec(X[:,2]);
v = vec(X[:,3]);
Y = readdlm(test, ',' , Int64);
xx = vec(Y[:,1]);
yy = vec(Y[:,2]);
vv = vec(Y[:,3]);
n = max(maximum(x), maximum(xx)); 
msize = max(maximum(y), maximum(yy));
println("Training dataset ", train, " and test dataset ", test, " are loaded. \n There are ", n, " users and ", msize, " items in the dataset.")
X = sparse(x, y, v, n, msize); # userid by movieid
Y = sparse(xx, yy, vv, n, msize);
# julia column major 
# now moveid by userid
X = X'; 
Y = Y'; 
rows = rowvals(X);
vals = nonzeros(X);
cols = zeros(Int, size(vals)[1]);
d2, d1 = size(X);
cc = 0;
cold_start_s = Set{Int64}()
for i = 1:d1
	tmp = nzrange(X, i);
	nowlen = size(tmp)[1];
	if nowlen < 11
		push!(cold_start_s, i)
	end
	for j = 1:nowlen
		cc += 1
		cols[cc] = i
	end
end

rows_t = rowvals(Y);
vals_t = nonzeros(Y);
cols_t = zeros(Int, size(vals_t)[1]);
cc = 0;
for i = 1:d1
	tmp = nzrange(Y, i);
	nowlen = size(tmp)[1];
	for j = 1:nowlen
		cc += 1
		cols_t[cc] = i
	end
end

U_mf = readdlm("W_bpr.txt");
U_mf = U_mf[:,2:size(U_mf)[2]]
U_mf = U_mf';
ndcg_k = 10
r = 100

V_tmp = readdlm("H_bpr.txt");
V_mf = zeros(r, d2);
zero_items = Set{Int64}();
for i = 1:size(V_tmp)[1]
    item_id = convert(Int64, V_tmp[i][1]);
    push!(zero_items, item_id);
    V_mf[:,item_id] = V_tmp[i,:][2: (r + 1)]
end



p1, p5, p10 = compute_precision(U_mf, V_mf, X, Y, d1, d2, rows, vals, rows_t, vals_t, zero_items)
println("top 1 precision: ", p1, " top 5 precision: ", p5, " top 10 precision: ", p10)

#outfile = "outfile_sse_bpr.dat"
# writing to files is very similar:
#f = open(outfile, "a")
# both print and println can be used as usual but with f as their first arugment
#println(f, "[", p1, ",", p5, ",", p10, "],")
#close(f)
