##환경
python 3.6
numpy 1.12.1
scipy 0.19.0
matplotlib 2.0.2
Windows 10, 64bit에서 아나콘다를 이용해 테스트 됨.

##구성
simulator 클래스에 세팅(포지션 별 관측 확률, 아이템 별 관측 시 클릭될 조건부 확률)을 준 뒤 run을 실행하면 해당 세팅에서 각 모델(PBM_TS, RandomPick 등)의 행동을 시뮬레이션 한 뒤 결과를 저장한다.

##MAB 클래스
* 기본적으로 MAB 모델이 얻을 수 있는 정보는 아이템의 아이디와 포지션 별 유저의 관측 확률뿐이다.

* MAB의 select_items 함수를 뽑아야 할 아이템의 개수를 인자로 넘겨주며 호출하면, MAB는 이때까지 얻은 정보를 토대로 아이템 아이디* 의 리스트를 반환해야 한다. 이 때 반환되는 리스트에서 아이템 아이디의 인덱스가 해당 아이템이 노출될 포지션을 뜻한다.

* MAB의 update 함수에 MAB가 select_items의 결과로 반환한 리스트와, 해당 리스트에 대한 유저 피드백 리스트(list of boolean, k번째 인덱스의 값이 True면 유저가 해당 위치의 아이템을 클릭한 것이고 False면 관측 후 무시했거나 관측하지 않은 것)를 인자로 호출하면 MAB는 내부 정보를 업데이트한다.

##클래스 변수 의미
* K : 가능한 아이템의 개수

* L : 가능한 포지션의 개수

* posProb : posProb[i]는 i번째 포지션을 유저가 관측할 확률. 가능한 인덱스는 [0, L-1]

* itemProb : itemProb[i]는 i번 아이템이 관측됐을 때 유저가 클릭할 확률. 가능한 인덱스는 [0, K-1]

* S : K * L 2차원 배열. S[k][l]은 k번째 아이템이 l번째 위치에 노출된 후 유저에게 클릭된 횟수이다.

* N : K * L 2차원 배열. N[k][l]은 k번째 아이템이 l번째 위치에 노출시킨 횟수이다(유저가 관측했는지 여부는 상관 없음).

* Nc : K * L 2차원 배열. N의 bias-corrected 버전이다.