 require 'nn'
require 'nngraph' 
 
 --- Count the number of pairs in a table
 function table_size(t)
  local a=0
  for k,_ in pairs(t) do
    a=a+1
  end
  return a
end


--- Compute an index of the users that appear at least one time in the training and testing cascades
function buildIndex(train, test)
  local count={}
  local nb=0
    
   print("\tReading "..train)
    for line in io.lines(train) do
      local tokens=string.gmatch(line,"[^%s]+")
      for token in tokens do
        iter=string.gmatch(token,"[^,]+")
        local user=iter()
        local timestamp=tonumber(iter())
        if (count[user]==nil) then count[user]=0 end
        count[user]=count[user]+1
      end
    end
  
    print("\tReading "..test)
    local countt={}    
    for line in io.lines(test) do
      local tokens=string.gmatch(line,"[^%s]+")
      for token in tokens do
        iter=string.gmatch(token,"[^,]+")
        local user=iter()
        local timestamp=tonumber(iter())
        if (countt[user]==nil) then countt[user]=0 end
        countt[user]=countt[user]+1
      end
    end
    
  local index={}
  local pos=1
  for u,n in pairs(count) do
    if (countt[u]~=nil) then index[u]=pos; pos=pos+1; end
  end
  print("\t"..(pos-1).." different users.")
  return index
end

----- Read a set of cascade from a file given a index of users
function readFromFile(filename,index_users)
  assert(index_users~=nil)
  local nb_users=0
  nb_users=table_size(index_users)  
   
  local cascades_users={}
  local cascades_timestamps={}
  local nb_cascades=1
    
  for line in io.lines(filename) do
    local sequence_users={}
    local sequence_timestamps={}
    local pos=1
    
    local tokens=string.gmatch(line,"[^%s]+")
    for token in tokens do
      iter=string.gmatch(token,"[^,]+")
      local user=iter()
      local timestamp=tonumber(iter())
            
      if (index_users[user]~=nil) then
          sequence_users[pos]=index_users[user]
          sequence_timestamps[pos]=timestamp
          pos=pos+1
      end
    end
    if (#sequence_users>1) then
      cascades_users[nb_cascades]=sequence_users
      cascades_timestamps[nb_cascades]=sequence_timestamps
      nb_cascades=nb_cascades+1      
    end
  end
  print("\tNb cascades = "..(nb_cascades-1).." for nb_users="..nb_users)
  local retour={cascades=cascades_users,timestamps=cascades_timestamps,index=index_users,nb_users=nb_users,nb_cascades=nb_cascades-1}
  retour.size_cascades={}
  for i=1,#retour.cascades do retour.size_cascades[i]=#(retour.cascades[i]) end
  return(retour)
end

function computeDistanceMatrix(zs,nb_users)
	print("Computing distance matrix of "..nb_users.." users.")
	local matrix=torch.Tensor(nb_users,nb_users):fill(0)
	local is=torch.Tensor(1):fill(1)
	local dist=nn.PairwiseDistance(2)
	for u=1,nb_users do
		local z1=zs[u]:forward(is)
		for u2=u,nb_users do
			local z2=zs[u2]:forward(is)
			matrix[u][u2]=dist:forward({z1,z2})[1]
			matrix[u2][u]=matrix[u][u2]
		end
	end
	return(matrix)
end

--- Compute the average precision over a cascade given a distanceMatrix
function computeAveragePrecision(cascade,size_cascade,distanceMatrix)
  local idx_source=cascade[1]
  local liste={}
  for u=1,distanceMatrix:size(1) do
    liste[u]={}
    liste[u].user=u
    liste[u].distance=distanceMatrix[idx_source][u]
  end
  for i=1,size_cascade do
    liste[cascade[i]].relevant=true
  end
  
  function compare(a,b)
    return(a.distance<b.distance)
  end
  table.sort(liste,compare)
  
  local nb_positive=0
  local rank=1
  local avgp=0
  while(nb_positive<size_cascade) do
    local elt=liste[rank]
    if(elt.relevant) then nb_positive=nb_positive+1; local pre=nb_positive/rank; avgp=avgp+pre end
    rank=rank+1
  end
  avgp=avgp/size_cascade
  return(avgp)  
end
  
  -- Compute the average precision over a cascade given a distanceMatrix
function computeMAP(cascades,size_cascades,distanceMatrix)
  local map=0
  for i=1,#size_cascades do
    map=map+computeAveragePrecision(cascades[i],size_cascades[i],distanceMatrix)
  end
  map=map/#size_cascades
  return map
end
  
  
	

-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------
-----------------------------------------------------------------------------------
cmd=torch.CmdLine()
cmd:text()
cmd:option('--training_cascades', "", 'training_cascades file')
cmd:option('--testing_cascades', "", 'testing_cascades file')
cmd:option('--outputFile', "", 'The outputfile where to store the embeddings of users')

cmd:option('--learningRate', 0.01, 'learning rate')
cmd:option('--maxEpoch', 1000, 'maximum number of epochs to run')
cmd:option('--evaluationEpoch', 10, 'Number of steps where evaluation is made')

cmd:option('--uniform', 0.1, 'initialize parameters using a gaussian distribution')
cmd:option('--N', 10, 'Dimension of the latent space')
cmd:text()
 
local opt = cmd:parse(arg or {})

print(opt)

print("Building Index of users (users that appear at least one time in both train and test files")
index_users=buildIndex(opt.training_cascades,opt.testing_cascades)

print("Reading training and testing cascades")
train_cascades=readFromFile(opt.training_cascades,index_users)
test_cascades=readFromFile(opt.testing_cascades,index_users)

print("Initalisation of the embeddings...")
--	print(train_cascades.index)
local zs={}
for u=1,train_cascades.nb_users do
	zs[u]=nn.Linear(1,opt.N)
	zs[u]:reset(opt.uniform)
end

local is=torch.Tensor(1):fill(1)
local criterion=nn.MarginRankingCriterion(1)


for iteration=1,opt.maxEpoch do
	-- Evaluaton
	if ((iteration-1)%opt.evaluationEpoch==0) then
		local distance_matrix=computeDistanceMatrix(zs,train_cascades.nb_users)		
    local train_map=computeMAP(train_cascades.cascades,train_cascades.size_cascades,distance_matrix)
    print("Training MAP = "..train_map)
    local testing_map=computeMAP(test_cascades.cascades,test_cascades.size_cascades,distance_matrix)
    print("Testing MAP = "..testing_map)
	end
		
	-- SGD 
	local total_loss=0
	for i=1,train_cascades.nb_cascades do
		local idx_cascade=math.random(train_cascades.nb_cascades)
		local cascade=train_cascades.cascades[idx_cascade]
		local size_cascade=train_cascades.size_cascades[idx_cascade]

		local user_source=cascade[1] -- source of the cascade
		local idx_contaminated=math.random(size_cascade-1)+1
		local user_contaminated=cascade[idx_contaminated] -- one user in the cascade, but not the source
		
		local user_further=math.random(train_cascades.nb_users) -- another user which is not between user_source and user_contaminated in the cascade
		local flag=true
		while(flag) do
			flag=false
			for i=1,idx_contaminated do if (cascade[i]==user_further) then flag=true end end
			if (flag) then user_further=math.random(train_cascades.nb_users) end
		end
		-- Building the modele
		local input=nn.Identity()()
		local z_source=zs[user_source](input)
		local z_contaminated=zs[user_contaminated](input)
		local z_further=zs[user_further](input)
		local d1=nn.PairwiseDistance(2)({z_source,z_contaminated})
		local d2=nn.PairwiseDistance(2)({z_source,z_further})
		local model=nn.gModule({input},{d1,d2})

		-- forward/backward
		model:zeroGradParameters()
		local out=model:forward(is)
	  local loss=criterion:forward(out,-1)
	  total_loss=total_loss+loss
	  local delta=criterion:backward(out,-1)
	  model:backward(is,delta)
	  model:updateParameters(opt.learningRate)
  end
  total_loss=total_loss/train_cascades.nb_cascades
  print("Average loss at iteration "..iteration .." is "..total_loss)
end

--- Save embeddings in a file
if (opt.outputFile~="") then
  print("Saving embeddings in "..opt.outputFile) 
  io.output(opt.outputFile)
  for u,idx in pairs(index_users) do
    local emb=zs[idx]:forward(is)
    io.write(u)
    for j=1,emb:size(1) do
      io.write(" "..emb[j])
    end
    io.write("\n")
  end
end





 
