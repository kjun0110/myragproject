export interface Match {
  id: string;
  homeTeam: string;
  awayTeam: string;
  date: string;
  time: string;
  venue: string;
  availableSeats: number;
  price: number;
}

export interface BettingOdds {
  matchId: string;
  homeWin: number;
  draw: number;
  awayWin: number;
  homeWinProbability: number;
  drawProbability: number;
  awayWinProbability: number;
}

export interface Product {
  id: string;
  name: string;
  type: "ticket" | "merchandise" | "experience";
  price: number;
  stock: number;
  description: string;
}

export interface Member {
  id: string;
  name: string;
  email: string;
  phone: string;
  membershipLevel: "bronze" | "silver" | "gold" | "platinum";
  joinDate: string;
  totalSpent: number;
}
