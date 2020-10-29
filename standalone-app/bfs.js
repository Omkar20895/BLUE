


L = 20;
var p = 0.3;
mode = 2;
zero_thresh = 0.05;





// Converts an edgelist to an adjacency list representation
// In this program, we use a dictionary as an adjacency list,
// where each key is a vertex, and each value is a list of all
// vertices adjacent to that vertex
var convert_edgelist_to_adjlist = function(edgelist) {
  var adjlist = {};
  var i, len, pair, u, v;
  for (i = 0, len = edgelist.length; i < len; i += 1) {
    pair = edgelist[i];
    u = pair[0];
    v = pair[1];
    if (adjlist[u]) {
      // append vertex v to edgelist of vertex u
      adjlist[u].push(v);
    } else {
      // vertex u is not in adjlist, create new adjacency list for it
      adjlist[u] = [v];
    }
    if (adjlist[v]) {
      adjlist[v].push(u);
    } else {
      adjlist[v] = [u];
    }
  }
  return adjlist;
};

function argMax(array) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}


var generate_lattice_adjlist = function(L) {
 var adjlist = new Array(L*L);
 var i,left,right,up,down;
 for(i =0; i<L*L; i+=1){
     adjlist[i]=[]
     if(i%L != 0)
         adjlist[i].push(i-1);
     if((i+1)%L != 0)
         adjlist[i].push(i+1);
     if( i - L >0)
         adjlist[i].push(i-L);
     if(i+L < L*L)
         adjlist[i].push(i+L);
 }
    return adjlist;

}

function BPQueue(N){
    this.length = 0;
    this.q = new Uint16Array(N);
}

BPQueue.prototype.pop = function(){
        return this.q[--this.length];
}

BPQueue.prototype.push = function(x){
        this.q[this.length ++] = x;
    
}

function GroupArray(L){
    this.L=L;
    this.udlr = parseInt("0000",2);
}

GroupArray.prototype = new Array;
GroupArray.prototype.add = function(idx){
    this.udlr |= get_up_down_left_right(idx,this.L);
    return this.push(idx);
}
GroupArray.prototype.update_udlr = function(){
    this.udlr = parseInt("0000",2);
    for(var i=0; i<this.length; i++){
        this.udlr |= get_up_down_left_right(this[i],this.L);
        if(this.udlr >= 15){
            break;
        }
    }

}

function BattlePerc(L){
    this.L=L;
    this.N = L*L;
    this.adjlist = generate_lattice_adjlist(L);
    this.status = new Uint8Array(this.N);
    this.status.fill(3);
    this.alive = new Uint8Array(this.N);
    this.alive.fill(1);
    this.visited = new Uint8Array(this.N);
    this.visited.fill(0);
    this.attacked = [];
    this.q = new BPQueue(this.N);
    this.mode = 0;
    this.zero_thresh=0.05;
    this.largest_non_alive_size=0;
    this.udlr_green_states = [parseInt("1111",2)]
    //uncomment for updown
    this.udlr_green_states = this.udlr_green_states.concat([parseInt("1110",2),parseInt("1101",2), parseInt("1100",2)])
    //uncomment for leftright
    this.udlr_green_states = this.udlr_green_states.concat([parseInt("0011",2),parseInt("0111",2),parseInt("1011",2),])
}


BattlePerc.prototype.set_mode = function(mode){
    this.mode = mode;
    if(this.mode == 2){
        this.largest_non_alive_size = this.zero_thresh * this.N;

    }
}

//Return 4bit int for OR-ing to see if component is touching any of four sides
function get_up_down_left_right(idx,L){
    up = idx<L? "1" : "0";
    down = idx>=(L-1)*L? "1" : "0";
    left = idx%L==0? "1" : "0";
    right = idx%L==L-1? "1" : "0"
    return parseInt(up + down + left + right,2)
}

// Breadth First Search using adjacency list
BattlePerc.prototype.bfs = function(v) {
  var current_group = new GroupArray(this.L);
  this.q.length=0;
  var i, len, adjV, nextVertex;

  this.q.push(v);
  var current_udlr =0;
  this.visited[v] = 1;
  while (this.q.length > 0) {
    v = this.q.pop();
    current_group.add(v);
    //current_udlr |= get_up_down_left_right(v);
    // Go through adjacency list of vertex v, and push any unvisited
    // vertex onto the queue.
    // This is more efficient than our earlier approach of going
    // through an edge list.
    adjV = this.adjlist[v];
    for (i = 0, len = adjV.length; i < len; i += 1) {
      nextVertex = adjV[i];
      if (!this.visited[nextVertex]) {

        this.q.push(nextVertex);
        this.visited[nextVertex] = 1;
      }
    }
  }
  return current_group;
};

// var pairs = [
//   ["a2", "a5"],
//   ["a3", "a6"],
//   ["a4", "a5"],
//   ["a7", "a9"]
// ];
// 
// var groups = [];
// var visited = {};
// var v;

// this should look like:
// {
//   "a2": ["a5"],
//   "a3": ["a6"],
//   "a4": ["a5"],
//   "a5": ["a2", "a4"],
//   "a6": ["a3"],
//   "a7": ["a9"],
//   "a9": ["a7"]
// }


BattlePerc.prototype.make_all_alive = function(){
for (var v=0; v<this.N; v++){
    this.alive[v] = 1;
}
}

BattlePerc.prototype.make_randomized_alive = function(p){
for (var v=0; v<this.N; v++){
    this.alive[v] = 1;
    if (Math.random() < p ){
        this.alive[v] = 0;
    }
}
}

BattlePerc.prototype.get_random_node_of_status = function(node_status){
    candidates=[];
    for(var v=0; v<this.N; v++){
        if(this.status[v] == node_status){
            candidates.push(v);
        }
    }
    //console.log(candidates.length)
    var cdidx = parseInt(Math.random() * candidates.length) % candidates.length;
    var choice = candidates[cdidx];

    return choice;

}

BattlePerc.prototype.get_components = function()
{
    this.groups = [];
    for (var v=0; v<this.N; v++) {
    if (!this.visited[v]) {
        this.groups.push(this.bfs(v));

    }
    }
}

BattlePerc.prototype.get_gcc_membership_vector = function(){
    

    var v;
    for(var v=0; v<this.N; v++){
        if (!this.alive[v]){
            this.visited[v] = 1;
        }
        else{
            this.visited[v] = 0;
    }
    }

    this.get_components();

    this.groups.sort(function(a,b){return b.length - a.length;});
    //console.log(groups);
    this.generate_status_vector_from_groups();

}

BattlePerc.prototype.generate_status_vector_from_groups = function(){
    var giant_idx = 0;


    for (var v=0; v<this.N; v++){
        if (!this.alive[v]){
            this.status[v] = 2;
        }
        else {
            this.status[v] = 1;
        }
    }
    

    switch(this.mode){
        case 0:
        
            if (this.groups.length > 1){
                this.largest_non_alive_size = Math.max(this.groups[1].length,this.largest_non_alive_size);
            }
        break;
        
        case 1:
            
            if (this.groups.length > 2){
                this.largest_non_alive_size = Math.max(this.groups[2].length,this.largest_non_alive_size);
            }
        break;
        
        case 2:
            
        break;

        case 3:
            
            for(var i=0; i<this.groups.length; i++){
                if (this.udlr_green_states.includes(this.groups[i].udlr)){
                    for(var v = 0; v<this.groups[i].length; v++){
                        this.status[this.groups[i][v]] = 0;
                    }
                }
            }
            return;
        break;
    }
    for( var i =0; i<this.groups.length; i++){
        if( this.groups[i].length <= this.largest_non_alive_size ){
            break;
        }
            for(var j=0; j<this.groups[i].length; j++){
                v = this.groups[i][j];
                this.status[v] = 0;
            }
         //i++;   
        }
}

BattlePerc.prototype.update_groups_after_click = function(idx){

    seeds = [];
    for (var i=0; i<this.adjlist[idx].length; i++){
        var v = this.adjlist[idx][i];
        if(this.status[ v ] == 0){
            seeds.push(v);
        }
    }
    var total_seed_count = parseInt(seeds.length);
    giant_groups = []
    for(var i=0; i<this.N; i++){
        this.visited[i] = this.status[i] == 0 ? 0 : 1;
    }
    this.visited[idx]=1;
    while( seeds.length > 0){
        var v = seeds.pop();
        if(this.visited[v]==1){
            continue;
        }
    
        var current_group = new GroupArray(this.L);
        this.q.length=0;
        var i, len, adjV, nextVertex,discovered_seed_index,found_seed_count=0;
      
        this.q.push(v);
        this.visited[v] = 1;
        while (this.q.length > 0) {
          v = this.q.pop();
          current_group.add(v);
          adjV = this.adjlist[v];
          discovered_seed_index = seeds.indexOf(v);
          if(discovered_seed_index != -1){
              found_seed_count++;
              //console.log(`Found ${v}`);
              if(found_seed_count == total_seed_count){
                  //console.log("stopping search")
                  break;
              }

          }
          for (var i = 0, len = adjV.length; i < len; i += 1) {
            nextVertex = adjV[i];
            if (!this.visited[nextVertex]) {
              this.q.push(nextVertex);
              this.visited[nextVertex] = 1;
            } 
          }
        }
        giant_groups.push(current_group);
    }
    //console.log(giant_groups)

    var original_group_idx;
        for(var i=0; i<this.groups.length; i++){
            if(this.groups[i].includes(idx)){
                original_group_idx = i;
                break;
            }
        }
    
    if (giant_groups.length <2){
        original_idx_in_group = this.groups[original_group_idx].indexOf(idx);
        this.groups[original_group_idx].splice(original_idx_in_group,1);
        if(get_up_down_left_right(idx) > 0){
            this.groups[original_group_idx].update_udlr();

        }
        //console.log("No change to groups, removing idx only")
        return false;
    } else {
        //console.log(`Groups changed, removing group ${original_group_idx} and adding ${giant_groups.length} back`)
        this.groups.splice(original_group_idx,1);
        this.groups = this.groups.concat(giant_groups);
        
        this.groups.sort(function(a,b){return b.length - a.length;});
        this.generate_status_vector_from_groups();
        return true;

    }
    



}

var get_cutoff_ids = function(status,oldstatus=[]){
    var i;
    cutoffs=[];
    if(oldstatus.length >0){
    for(i=0; i<L*L;i+=1){
        if(status[i] == 1 & oldstatus[i]!=1){
            cutoffs.push(i);
        }
    }
    } else{
    for(i=0; i<L*L;i+=1){
        if(status[i] == 1){
            cutoffs.push(i);
        }
    }
    }
    return cutoffs;
}

BattlePerc.prototype.attack_index = function(idx){
    this.alive[idx]= 0;
    var oldstatus = Uint8Array.from(this.status);
    this.get_gcc_membership_vector();
    return get_cutoff_ids(this.status,oldstatus);

}

BattlePerc.prototype.attack_index_fast = function(idx){
    if(this.alive[idx] == 0){
        return [];
    }
    this.alive[idx]=0;
    if(this.status[idx]==1){
        this.status[idx]=2;
        return [];    
    }
    this.status[idx]=2;
    var oldstatus = Uint8Array.from(this.status);
    if(this.update_groups_after_click(idx))
        return get_cutoff_ids(this.status,oldstatus);
    else
        return [];

}

BattlePerc.prototype.isClickable = function(idx){
    return true;
}

BattlePerc.prototype.toggle_index = function(idx){
    if(this.isClickable(idx)){
        if(this.alive[idx] == 1 ){
            this.attacked.push(idx);
            this.alive[idx] = 0;
            this.status[idx] = 2;
            this.update_groups_after_click(idx);
            var sacrificial_idx = this.get_random_node_of_status(2);
            //this.toggle_index(sacrificial_idx);
            console.log(sacrificial_idx);
        } else {
            this.alive[idx] = 1;
            this.status[idx] = 0;
            this.get_gcc_membership_vector();
        }
    } 

}

bp = new BattlePerc(L);
bp.zero_thresh=zero_thresh;
bp.set_mode(mode);
bp.make_randomized_alive(p);
alive = bp.alive;
// var status = get_gcc_membership_vector(adjlist,alive);
// var cutoff = get_cutoff_ids(status);

