
from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Any
import heapq

class Building:
    def __init__(self, bid: int, name: str, location: str):
        self.building_id = bid
        self.name = name
        self.location = location

    def __lt__(self, other):
        return self.building_id < other.building_id

    def __repr__(self):
        return f"Building(ID={self.building_id}, Name='{self.name}', Loc='{self.location}')"

class BSTNode:
    def __init__(self, building: Building):
        self.building = building
        self.left: Optional[BSTNode] = None
        self.right: Optional[BSTNode] = None

class BST:
    def __init__(self):
        self.root: Optional[BSTNode] = None

    def insert(self, building: Building):
        def _insert(node: Optional[BSTNode], b: Building) -> BSTNode:
            if node is None:
                return BSTNode(b)
            if b.building_id < node.building.building_id:
                node.left = _insert(node.left, b)
            elif b.building_id > node.building.building_id:
                node.right = _insert(node.right, b)
            else:
        
                node.building = b
            return node
        self.root = _insert(self.root, building)

    def search(self, bid: int) -> Optional[Building]:
        node = self.root
        while node:
            if bid == node.building.building_id:
                return node.building
            elif bid < node.building.building_id:
                node = node.left
            else:
                node = node.right
        return None

    def inorder(self) -> List[Building]:
        res: List[Building] = []
        def _in(node):
            if not node: return
            _in(node.left); res.append(node.building); _in(node.right)
        _in(self.root)
        return res

    def preorder(self) -> List[Building]:
        res: List[Building] = []
        def _pre(node):
            if not node: return
            res.append(node.building); _pre(node.left); _pre(node.right)
        _pre(self.root)
        return res

    def postorder(self) -> List[Building]:
        res: List[Building] = []
        def _post(node):
            if not node: return
            _post(node.left); _post(node.right); res.append(node.building)
        _post(self.root)
        return res

    def height(self) -> int:
        def _h(node):
            if not node: return 0
            return 1 + max(_h(node.left), _h(node.right))
        return _h(self.root)

class AVLNode:
    def __init__(self, building: Building):
        self.building = building
        self.left: Optional[AVLNode] = None
        self.right: Optional[AVLNode] = None
        self.height: int = 1

class AVLTree:
    def __init__(self):
        self.root: Optional[AVLNode] = None

    def get_height(self, node: Optional[AVLNode]) -> int:
        return node.height if node else 0

    def update_height(self, node: AVLNode):
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))

    def get_balance(self, node: Optional[AVLNode]) -> int:
        if not node: return 0
        return self.get_height(node.left) - self.get_height(node.right)

    def rotate_right(self, y: AVLNode) -> AVLNode:
        x = y.left
        T2 = x.right
    
        x.right = y
        y.left = T2
   
        self.update_height(y)
        self.update_height(x)
        return x

    def rotate_left(self, x: AVLNode) -> AVLNode:
        y = x.right
        T2 = y.left
      
        y.left = x
        x.right = T2
       
        self.update_height(x)
        self.update_height(y)
        return y

    def insert(self, building: Building):
        def _insert(node: Optional[AVLNode], b: Building) -> AVLNode:
            if not node:
                return AVLNode(b)
            if b.building_id < node.building.building_id:
                node.left = _insert(node.left, b)
            elif b.building_id > node.building.building_id:
                node.right = _insert(node.right, b)
            else:
                node.building = b
                return node

            self.update_height(node)
            balance = self.get_balance(node)

            if balance > 1 and b.building_id < node.left.building.building_id:
                return self.rotate_right(node)
          
            if balance < -1 and b.building_id > node.right.building.building_id:
                return self.rotate_left(node)
        
            if balance > 1 and b.building_id > node.left.building.building_id:
                node.left = self.rotate_left(node.left)
                return self.rotate_right(node)
      
            if balance < -1 and b.building_id < node.right.building.building_id:
                node.right = self.rotate_right(node.right)
                return self.rotate_left(node)

            return node

        self.root = _insert(self.root, building)

    def inorder(self) -> List[Building]:
        res: List[Building] = []
        def _in(node: Optional[AVLNode]):
            if not node: return
            _in(node.left); res.append(node.building); _in(node.right)
        _in(self.root)
        return res

    def height(self) -> int:
        return self.get_height(self.root)

class ExprNode:
    def __init__(self, val: str, left: Optional[ExprNode]=None, right: Optional[ExprNode]=None):
        self.val = val
        self.left = left
        self.right = right

def build_expr_tree_from_postfix(tokens: List[str]) -> Optional[ExprNode]:
    stack: List[ExprNode] = []
    operators = set(['+', '-', '*', '/', '^'])
    for t in tokens:
        if t not in operators:
            stack.append(ExprNode(t))
        else:
         
            r = stack.pop()
            l = stack.pop()
            stack.append(ExprNode(t, l, r))
    return stack[-1] if stack else None

def eval_expr_tree(node: ExprNode, variables: Dict[str, float]={}) -> float:
    if node.left is None and node.right is None:
      
        try:
            return float(node.val)
        except ValueError:
            return float(variables.get(node.val, 0.0))
    l = eval_expr_tree(node.left, variables)
    r = eval_expr_tree(node.right, variables)
    if node.val == '+': return l + r
    if node.val == '-': return l - r
    if node.val == '*': return l * r
    if node.val == '/': return l / r
    if node.val == '^': return l ** r
    raise ValueError("Unknown operator " + node.val)

class CampusGraph:
    def __init__(self):
       
        self.buildings: Dict[int, Building] = {}
        
        self.adj_list: Dict[int, List[Tuple[int, float]]] = {}
       
        self.adj_matrix: List[List[float]] = []
        self.max_bid_index_map: Dict[int,int] = {}  

    def add_building(self, building: Building):
        self.buildings[building.building_id] = building
        if building.building_id not in self.adj_list:
            self.adj_list[building.building_id] = []

    def add_edge(self, b1: int, b2: int, weight: float=1.0, undirected: bool=True):
        if b1 not in self.adj_list or b2 not in self.adj_list:
            raise KeyError("Both buildings must be added before connecting them")
        self.adj_list[b1].append((b2, weight))
        if undirected:
            self.adj_list[b2].append((b1, weight))

    def build_matrix(self):
     
        bids = sorted(self.buildings.keys())
        self.max_bid_index_map = {bid: i for i, bid in enumerate(bids)}
        n = len(bids)
        INF = float('inf')
        mat = [[INF]*n for _ in range(n)]
        for i in range(n):
            mat[i][i] = 0.0
        for u, edges in self.adj_list.items():
            ui = self.max_bid_index_map[u]
            for v, w in edges:
                vi = self.max_bid_index_map[v]
                mat[ui][vi] = w
        self.adj_matrix = mat
        return mat

    def bfs(self, start_bid: int) -> List[int]:
        visited = set()
        q = [start_bid]
        order = []
        visited.add(start_bid)
        while q:
            u = q.pop(0)
            order.append(u)
            for v, _ in self.adj_list.get(u, []):
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        return order

    def dfs(self, start_bid: int) -> List[int]:
        visited = set()
        order = []
        def _dfs(u):
            visited.add(u)
            order.append(u)
            for v, _ in self.adj_list.get(u, []):
                if v not in visited:
                    _dfs(v)
        _dfs(start_bid)
        return order

    def dijkstra(self, src: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        dist = {bid: float('inf') for bid in self.buildings}
        parent = {bid: None for bid in self.buildings}
        dist[src] = 0.0
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]: continue
            for v, w in self.adj_list.get(u, []):
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(heap, (nd, v))
        return dist, parent

    def shortest_path(self, src: int, dest: int) -> Tuple[List[int], float]:
        dist, parent = self.dijkstra(src)
        if dist[dest] == float('inf'):
            return [], float('inf')
        path = []
        cur = dest
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path, dist[dest]

    def kruskal_mst(self) -> Tuple[List[Tuple[int,int,float]], float]:
     
        edges = []
        seen = set()
        for u, neighbors in self.adj_list.items():
            for v, w in neighbors:
                if (v,u) in seen: continue
                seen.add((u,v))
                edges.append((w, u, v))
        edges.sort(key=lambda x: x[0])

        parent = {}
        rank = {}
        def make_set(x):
            parent[x] = x; rank[x] = 0
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x,y):
            rx, ry = find(x), find(y)
            if rx == ry: return False
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx; rank[rx] += 1
            return True

        for bid in self.buildings:
            make_set(bid)

        mst = []
        total_weight = 0.0
        for w,u,v in edges:
            if union(u,v):
                mst.append((u,v,w))
                total_weight += w
        return mst, total_weight


def demo():
   
    bdata = [
        (1, "Admin Block", "Central"),
        (2, "Library", "North Wing"),
        (3, "CSE Dept", "Block A"),
        (4, "ECE Dept", "Block B"),
        (5, "Hostel", "East Side"),
        (6, "Cafeteria", "South Wing")
    ]
    print("=== Building Data ===")
    buildings = [Building(*t) for t in bdata]
    for b in buildings:
        print(b)
    print()

    bst = BST()
    avl = AVLTree()
    for b in buildings:
        bst.insert(b)
        avl.insert(b)

    print("BST inorder:", bst.inorder())
    print("BST preorder:", bst.preorder())
    print("BST postorder:", bst.postorder())
    print("BST height:", bst.height())
    print()
    print("AVL inorder:", avl.inorder())
    print("AVL height:", avl.height())
    print()

    graph = CampusGraph()
    for b in buildings:
        graph.add_building(b)
  
    graph.add_edge(1,2,2.0)
    graph.add_edge(1,3,3.0)
    graph.add_edge(2,3,1.0)
    graph.add_edge(2,4,4.0)
    graph.add_edge(3,4,2.5)
    graph.add_edge(3,5,6.0)
    graph.add_edge(4,6,3.0)
    graph.add_edge(5,6,4.5)

    print("Adjacency List:")
    for k,v in graph.adj_list.items():
        print(k, "->", v)
    print()

    mat = graph.build_matrix()
    print("Adjacency Matrix (rows/cols are building IDs sorted):")
    bids_sorted = sorted(graph.buildings.keys())
    print("IDs:", bids_sorted)
    for row in mat:
        print(row)
    print()

    print("BFS from Building 1:", graph.bfs(1))
    print("DFS from Building 1:", graph.dfs(1))
    print()

    path, dist = graph.shortest_path(1,5)
    print(f"Shortest path 1 -> 5: {path} with distance {dist}")
    print()

    mst, total = graph.kruskal_mst()
    print("Kruskal MST edges (u,v,weight):", mst)
    print("Total MST weight:", total)
    print()

    tokens = ["base","units","rate","*","+"]
    expr_root = build_expr_tree_from_postfix(tokens)
    variables = {"base": 50, "units": 120, "rate": 0.6}
    cost = eval_expr_tree(expr_root, variables)
    print("Expression tree eval (energy bill) =>", cost)

if __name__ == "__main__":
    demo()
