import { ChangeDetectionStrategy, Component, Input, OnInit } from '@angular/core';

@Component({
  selector: 'app-notification',
  templateUrl: './notification.component.html',
  styleUrls: ['./notification.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class NotificationComponent implements OnInit {
  @Input() public show: boolean;
  @Input() public type: string;
  @Input() public title: string;
  @Input() public description: string;
  @Input() public position: string;
  constructor() { }

  ngOnInit() {
  }

  hide = () => this.show = false;
}
