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

----- Read a set of cascade from a file given a index of users
function loadEmbeddings(filename)
  local zs={}  
  local index={}
  local pos=1
  for line in io.lines(filename) do
    local tokens=string.gmatch(line,"[^%s]+")
    local dim=0
    local v={}
    local user=tokens()
    for token in tokens do
      dim=dim+1
      v[dim]=tonumber(token)      
    end
    local z=torch.Tensor(dim)
    for i=1,dim do z[i]=v[i] end
    index[user]=pos
    zs[pos]=z    
    pos=pos+1
  end  
  return({zs,index})
end

function computeDistanceMatrix(zs,nb_users)
	print("Computing distance matrix of "..nb_users.." users.")
	local matrix=torch.Tensor(nb_users,nb_users):fill(0)
	local is=torch.Tensor(1):fill(1)
	local dist=nn.PairwiseDistance(2)
	for u=1,nb_users do
		local z1=zs[u]
		for u2=u,nb_users do
			local z2=zs[u2]
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
cmd:option('--cascades', "", 'cascades file')
cmd:option('--embeddings', "", 'embeddings file')
cmd:text()

local opt = cmd:parse(arg or {})

print("Loading embeddings")

t=loadEmbeddings(opt.embeddings)
zs=t[1]
index_users=t[2]

print(#zs.." users found in dimension "..zs[1]:size(1))
cascades=readFromFile(opt.cascades,index_users)

		local distance_matrix=computeDistanceMatrix(zs,cascades.nb_users)		
    print("Computing MAP")
    local map=computeMAP(cascades.cascades,cascades.size_cascades,distance_matrix)
    print("MAP = "..map)




 
